"""Distributional + stability + latent-class analyses of resilient vs vulnerable subjects.

Three independent analyses, each grounded in a different field:

1. ``wasserstein_per_celltype`` — optimal transport (math/stats). Per cell type,
   compute the 1D Wasserstein-1 distance between the resilient and vulnerable
   groups' per-subject expression for each gene. Captures distributional shift.

2. ``stability_selection`` — stability-selection (Meinshausen & Bühlmann 2010).
   Repeatedly subsample subjects and run a per-gene effect-size statistic
   (Wilcoxon's rank-biserial r); retain genes selected in ≥ pi_thr of
   resamples. Optionally sweep over a grid of rb_thresholds and report the
   selection probability surface.

3. ``latent_class_on_residuals`` — latent class analysis (psychometrics).
   Fit a Gaussian mixture on the per-subject residual ``y - ŷ`` and select
   the number of components by BIC + AIC (both reported). Tests whether
   resilience is better modeled as a continuum or discrete mixture.

All functions are pure: numpy in, plain dicts out (JSON-serializable).
"""
from __future__ import annotations

import logging
from typing import Sequence

logger = logging.getLogger(__name__)

import numpy as np
from scipy.stats import wasserstein_distance, rankdata
from sklearn.mixture import GaussianMixture


def wasserstein_per_celltype(
    expression_per_subject: np.ndarray,
    is_resilient: np.ndarray,
    cell_type_names: Sequence[str] | None = None,
    gene_names: Sequence[str] | None = None,
) -> dict:
    """Per-cell-type per-gene Wasserstein-1 distance, resilient vs vulnerable.

    Parameters
    ----------
    expression_per_subject
        Array ``(n_subjects, n_celltypes, n_genes)`` of per-subject mean
        expression per (cell type, gene). May contain NaN.
    is_resilient
        Boolean ``(n_subjects,)``.
    cell_type_names
        Optional names; default ``CT_<i>``.
    gene_names
        Optional names; default ``gene_<j>``.

    Returns
    -------
    dict
        Per cell type: mean per-gene W-1 + top 10 by W-1.
    """
    n_subj, n_ct, n_gene = expression_per_subject.shape
    is_res = np.asarray(is_resilient, dtype=bool)
    if cell_type_names is None:
        cell_type_names = [f"CT_{i}" for i in range(n_ct)]
    if gene_names is None:
        gene_names = [f"gene_{j}" for j in range(n_gene)]

    per_ct = []
    for ct in range(n_ct):
        ct_data = expression_per_subject[:, ct, :]
        per_gene_w = np.full(n_gene, np.nan, dtype=np.float64)
        for g in range(n_gene):
            res = ct_data[is_res, g]
            vul = ct_data[~is_res, g]
            res = res[np.isfinite(res)]
            vul = vul[np.isfinite(vul)]
            if res.size < 2 or vul.size < 2:
                continue
            per_gene_w[g] = wasserstein_distance(res, vul)
        finite = per_gene_w[np.isfinite(per_gene_w)]
        mean_w = float(finite.mean()) if finite.size else float("nan")
        order = np.argsort(per_gene_w)[::-1]
        top10 = []
        for idx in order[:10]:
            if np.isfinite(per_gene_w[idx]):
                top10.append((str(gene_names[idx]), float(per_gene_w[idx])))
        per_ct.append({
            "cell_type": str(cell_type_names[ct]),
            "wasserstein_per_gene_mean": mean_w,
            "wasserstein_per_gene_top10": top10,
        })

    return {
        "n_resilient": int(is_res.sum()),
        "n_vulnerable": int((~is_res).sum()),
        "per_cell_type": per_ct,
    }


def _rank_biserial_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Rank-biserial correlation between two groups (Wilcoxon effect size).

    Equivalent to (2*U / (n_x * n_y)) - 1 where U is Mann-Whitney U.
    Bounded in [-1, +1].

    Note: kept private here for backward compatibility with existing tests;
    public alias lives at ``de_resilience.rank_biserial_correlation``.
    """
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if x.size < 1 or y.size < 1:
        return float("nan")
    pooled = np.concatenate([x, y])
    ranks = rankdata(pooled)
    rank_x = ranks[: x.size]
    U = float(rank_x.sum() - x.size * (x.size + 1) / 2.0)
    return float(2.0 * U / (x.size * y.size) - 1.0)


def _vectorized_rank_biserial(
    x_resampled: np.ndarray, y_resampled: np.ndarray,
) -> np.ndarray:
    """Vectorized rank-biserial across all genes for one resample.

    x_resampled: (n_x, n_features); y_resampled: (n_y, n_features).
    Returns: (n_features,) rank-biserial per feature.
    """
    n_x = x_resampled.shape[0]
    n_y = y_resampled.shape[0]
    pooled = np.vstack([x_resampled, y_resampled])  # (n_x + n_y, n_features)
    # Per-column ranks via scipy.
    ranks = rankdata(pooled, axis=0)
    rank_x_sum = ranks[:n_x].sum(axis=0)
    U = rank_x_sum - n_x * (n_x + 1) / 2.0
    return 2.0 * U / (n_x * n_y) - 1.0


def stability_selection(
    expression_matrix: np.ndarray,
    is_resilient: np.ndarray,
    *,
    n_bootstrap: int = 100,
    subsample_frac: float = 0.5,
    rb_threshold: float = 0.2,
    rb_threshold_path: Sequence[float] | None = None,
    pi_threshold: float = 0.8,
    seed: int = 42,
    gene_names: Sequence[str] | None = None,
) -> dict:
    """Stability selection: genes that pass a |rank-biserial| cutoff in ≥ pi_thr resamples.

    The default ``rb_threshold=0.2`` is a heuristic ("moderate effect" per
    Cohen-style guidance for rank correlations); set ``rb_threshold_path``
    to sweep the threshold and report the full selection-probability surface.

    Vectorized inner loop: per resample, computes per-gene rank-biserial in a
    single ``scipy.stats.rankdata`` call across all features (~100x faster
    than the per-gene Python loop in the original implementation).

    Parameters
    ----------
    expression_matrix
        Shape ``(n_subjects, n_features)``.
    is_resilient
        Boolean ``(n_subjects,)``.
    n_bootstrap
        Number of resamples (default 100).
    subsample_frac
        Fraction of each group to sample per resample (default 0.5).
    rb_threshold
        |rank-biserial| threshold. Used when ``rb_threshold_path`` is None.
    rb_threshold_path
        If not None, sweep these thresholds and return the full probability
        matrix in ``selection_probability_path`` (shape n_thresholds x n_features).
    pi_threshold
        Proportion of resamples at which a gene is "stable" (default 0.8).
    seed
        RNG seed.
    gene_names
        Optional names.

    Returns
    -------
    dict with ``selection_probability``, ``stable_indices``, ``stable_genes``,
    optionally ``selection_probability_path``, plus ``config``.
    """
    rng = np.random.default_rng(seed)
    is_res = np.asarray(is_resilient, dtype=bool)
    res_idx = np.flatnonzero(is_res)
    vul_idx = np.flatnonzero(~is_res)
    n_res_sub = max(2, int(len(res_idx) * subsample_frac))
    n_vul_sub = max(2, int(len(vul_idx) * subsample_frac))
    n_features = expression_matrix.shape[1]

    if rb_threshold_path is None:
        thresholds = (float(rb_threshold),)
    else:
        thresholds = tuple(float(t) for t in rb_threshold_path)
    n_thr = len(thresholds)

    counts = np.zeros((n_thr, n_features), dtype=np.int64)
    for _ in range(n_bootstrap):
        ri = rng.choice(res_idx, size=n_res_sub, replace=False)
        vi = rng.choice(vul_idx, size=n_vul_sub, replace=False)
        x_res_b = expression_matrix[ri]
        x_vul_b = expression_matrix[vi]
        rb = _vectorized_rank_biserial(x_res_b, x_vul_b)
        for ti, t in enumerate(thresholds):
            counts[ti] += (np.abs(rb) >= t).astype(np.int64)

    probs = counts / float(n_bootstrap)
    if gene_names is None:
        gene_names = [f"gene_{j}" for j in range(n_features)]

    # Use the configured rb_threshold as the canonical "stable" set. Match
    # by closest floating-point value (np.argmin of |Δ|) instead of exact
    # equality, so CLI roundtrips of e.g. 0.2 don't silently fall back to
    # index 0 on rounding error.
    thresholds_arr = np.asarray(thresholds, dtype=np.float64)
    canonical_idx = int(np.argmin(np.abs(thresholds_arr - float(rb_threshold))))
    if not np.isclose(thresholds_arr[canonical_idx], float(rb_threshold), rtol=1e-9, atol=1e-12):
        raise ValueError(
            f"rb_threshold={rb_threshold} not in thresholds path {thresholds}; "
            f"closest is {thresholds_arr[canonical_idx]}"
        )
    stable = np.flatnonzero(probs[canonical_idx] >= pi_threshold)

    result = {
        "selection_probability": probs[canonical_idx].tolist(),
        "stable_indices": [int(i) for i in stable],
        "stable_genes": [str(gene_names[i]) for i in stable],
        "config": {
            "n_bootstrap": int(n_bootstrap),
            "subsample_frac": float(subsample_frac),
            "rb_threshold": float(rb_threshold),
            "pi_threshold": float(pi_threshold),
            "seed": int(seed),
            "n_resilient": int(len(res_idx)),
            "n_vulnerable": int(len(vul_idx)),
        },
    }
    if rb_threshold_path is not None:
        result["selection_probability_path"] = probs.tolist()
        result["thresholds"] = list(thresholds)
    return result


def latent_class_on_residuals(
    residuals: np.ndarray,
    *,
    k_max: int = 5,
    n_init: int = 10,
    seed: int = 42,
    covariance_type: str = "full",
) -> dict:
    """Fit Gaussian mixtures on residuals; report BIC + AIC, choose K by BIC.

    Returns both BIC and AIC paths so callers can apply alternative criteria.

    Parameters
    ----------
    residuals
        1D ``(n_subjects,)`` of per-subject ``y - ŷ``.
    k_max
        Max number of components (1..k_max).
    n_init
        EM restarts per K.
    seed
        RNG seed.
    covariance_type
        ``"full"`` (default), ``"diag"``, ``"tied"``, ``"spherical"``.
        For 1D data all are equivalent; ``"full"`` is the conventional choice.

    Returns
    -------
    dict with ``best_k``, ``bic_per_k``, ``aic_per_k``, ``is_unimodal``,
    ``best_model_means/stds/weights/assignments``, ``config``.
    """
    r = np.asarray(residuals, dtype=np.float64).reshape(-1, 1)
    finite_mask = np.isfinite(r.ravel())
    n_finite = int(finite_mask.sum())
    if n_finite < 2:
        raise ValueError(
            f"Need at least 2 finite residuals to fit any GMM; got {n_finite}"
        )
    r_fit = r[finite_mask]
    # If fewer subjects than requested k_max, cap k at n_finite-1 (every
    # GMM component needs at least 2 samples to estimate variance).
    k_eff = min(int(k_max), max(1, n_finite - 1))
    if k_eff < int(k_max):
        logger.warning(
            "k_max=%d exceeds n_finite-1=%d; capping K to %d for stability.",
            k_max, n_finite - 1, k_eff,
        )

    bics, aics, fitted = [], [], []
    for k in range(1, k_eff + 1):
        gmm = GaussianMixture(
            n_components=k, n_init=n_init, random_state=seed,
            covariance_type=covariance_type,
        )
        gmm.fit(r_fit)
        bics.append(float(gmm.bic(r_fit)))
        aics.append(float(gmm.aic(r_fit)))
        fitted.append(gmm)
    best_k_idx = int(np.argmin(bics))
    best = fitted[best_k_idx]
    best_k = best_k_idx + 1

    assignments = np.full(r.size, -1, dtype=np.int64)
    if finite_mask.any():
        labels = best.predict(r_fit)
        assignments[finite_mask] = labels

    means = best.means_.ravel().tolist()
    if covariance_type == "full":
        stds = np.sqrt(np.array([c[0, 0] for c in best.covariances_])).tolist()
    elif covariance_type in ("diag", "spherical"):
        stds = np.sqrt(best.covariances_.ravel()).tolist()
    elif covariance_type == "tied":
        stds = [float(np.sqrt(best.covariances_[0, 0]))] * best_k
    else:
        stds = np.sqrt(np.atleast_1d(best.covariances_).ravel()).tolist()
    weights = best.weights_.tolist()
    order = np.argsort(means)
    means_sorted = [float(means[i]) for i in order]
    stds_sorted = [float(stds[i]) for i in order]
    weights_sorted = [float(weights[i]) for i in order]
    relabel = {old: new for new, old in enumerate(order)}
    assignments_sorted = np.array(
        [relabel[a] if a != -1 else -1 for a in assignments], dtype=np.int64,
    )

    return {
        "best_k": best_k,
        "bic_per_k": bics,
        "aic_per_k": aics,
        "is_unimodal": bool(best_k == 1),
        "best_model_means": means_sorted,
        "best_model_stds": stds_sorted,
        "best_model_weights": weights_sorted,
        "best_model_assignments": assignments_sorted.tolist(),
        "config": {
            "k_max": int(k_max),
            "n_init": int(n_init),
            "seed": int(seed),
            "covariance_type": str(covariance_type),
        },
    }
