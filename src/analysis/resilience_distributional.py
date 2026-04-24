"""Distributional + stability + latent-class analyses of resilient vs vulnerable subjects.

Three independent analyses, each grounded in a different field:

1. ``wasserstein_per_celltype`` — optimal transport (math/stats). Per cell type,
   compute the 1D Wasserstein-2 distance between the resilient and vulnerable
   groups' average expression vectors per subject (gene-by-gene). Captures
   distributional shift, not just mean shift.

2. ``stability_selection`` — stability-selection (machine learning theory).
   Repeatedly subsample subjects and run a per-gene effect-size statistic
   (Wilcoxon's rank-biserial r); retain genes selected in ≥ pi_thr of
   resamples. Controls FDR robustly without heavy multiple-test correction.

3. ``latent_class_on_residuals`` — latent class analysis (psychometrics).
   Fit a Gaussian mixture on the per-subject residual ``y - ŷ`` and select
   the number of components by BIC. Tests whether resilience is better
   modeled as a continuum or a discrete mixture.

All functions are pure: they take numpy arrays + small config and return
plain dicts with results, suitable for JSON serialization.
"""
from __future__ import annotations

from typing import Sequence

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
        Array of shape ``(n_subjects, n_celltypes, n_genes)`` holding
        per-subject mean expression per (cell type, gene). May contain NaN
        for missing (subject, cell type) combinations.
    is_resilient
        Boolean array of shape ``(n_subjects,)``; True = resilient,
        False = vulnerable.
    cell_type_names
        Optional names for each cell type axis. Defaults to ``CT_<i>``.
    gene_names
        Optional names for each gene axis. Defaults to ``gene_<j>``.

    Returns
    -------
    dict
        ``{
            "n_resilient": int,
            "n_vulnerable": int,
            "per_cell_type": [
                {
                    "cell_type": str,
                    "wasserstein_per_gene_mean": float,
                    "wasserstein_per_gene_top10": [(gene, distance), ...],
                },
                ...
            ],
        }``

        For each cell type, we report (a) the mean per-gene Wasserstein
        distance averaged across genes (a scalar shift summary) and
        (b) the top 10 genes by per-gene Wasserstein distance (the genes
        most distributionally different between groups).
    """
    n_subj, n_ct, n_gene = expression_per_subject.shape
    is_res = np.asarray(is_resilient, dtype=bool)
    if cell_type_names is None:
        cell_type_names = [f"CT_{i}" for i in range(n_ct)]
    if gene_names is None:
        gene_names = [f"gene_{j}" for j in range(n_gene)]

    per_ct = []
    for ct in range(n_ct):
        # Per-gene Wasserstein-1 between resilient and vulnerable distributions
        # of the (subject, gene) means for THIS cell type.
        ct_data = expression_per_subject[:, ct, :]  # (n_subj, n_gene)
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
        # Top 10 genes by Wasserstein distance.
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

    Equivalent to (2*U / (n_x * n_y)) - 1 where U is the Mann-Whitney U
    statistic. Bounded in [-1, +1]; sign indicates direction.
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


def stability_selection(
    expression_matrix: np.ndarray,
    is_resilient: np.ndarray,
    *,
    n_bootstrap: int = 100,
    subsample_frac: float = 0.5,
    rb_threshold: float = 0.2,
    pi_threshold: float = 0.8,
    seed: int = 42,
    gene_names: Sequence[str] | None = None,
) -> dict:
    """Stability selection: genes that pass an effect-size cutoff in ≥ pi_thr of resamples.

    Parameters
    ----------
    expression_matrix
        Shape ``(n_subjects, n_features)`` (e.g., per-subject pseudobulk
        flattened across cell types or per cell type).
    is_resilient
        Boolean array ``(n_subjects,)``.
    n_bootstrap
        Number of resamples (default 100).
    subsample_frac
        Fraction of each group to sample per resample (default 0.5).
    rb_threshold
        |rank-biserial correlation| threshold to call a gene "selected"
        within a single resample (default 0.2 — moderate effect).
    pi_threshold
        Proportion of resamples in which a gene must be selected to be
        retained (default 0.8 — Meinshausen & Bühlmann's recommendation).
    seed
        RNG seed.
    gene_names
        Optional names for the n_features axis.

    Returns
    -------
    dict
        ``{
            "selection_probability": [...n_features],
            "stable_indices": [...indices with prob >= pi_thr],
            "stable_genes": [...names with prob >= pi_thr],
            "config": {...}
        }``
    """
    rng = np.random.default_rng(seed)
    is_res = np.asarray(is_resilient, dtype=bool)
    res_idx = np.flatnonzero(is_res)
    vul_idx = np.flatnonzero(~is_res)
    n_res_sub = max(2, int(len(res_idx) * subsample_frac))
    n_vul_sub = max(2, int(len(vul_idx) * subsample_frac))
    n_features = expression_matrix.shape[1]

    selection_count = np.zeros(n_features, dtype=np.int64)
    for _ in range(n_bootstrap):
        ri = rng.choice(res_idx, size=n_res_sub, replace=False)
        vi = rng.choice(vul_idx, size=n_vul_sub, replace=False)
        x_res = expression_matrix[ri, :]
        x_vul = expression_matrix[vi, :]
        for g in range(n_features):
            rb = _rank_biserial_correlation(x_res[:, g], x_vul[:, g])
            if abs(rb) >= rb_threshold:
                selection_count[g] += 1

    selection_prob = selection_count / float(n_bootstrap)
    stable = np.flatnonzero(selection_prob >= pi_threshold)
    if gene_names is None:
        gene_names = [f"gene_{j}" for j in range(n_features)]

    return {
        "selection_probability": selection_prob.tolist(),
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


def latent_class_on_residuals(
    residuals: np.ndarray,
    *,
    k_max: int = 5,
    n_init: int = 10,
    seed: int = 42,
) -> dict:
    """Fit Gaussian mixtures on residuals, choose K by BIC.

    Parameters
    ----------
    residuals
        1D array ``(n_subjects,)`` of per-subject ``y - ŷ``.
    k_max
        Maximum number of components to consider (1..k_max).
    n_init
        Number of EM restarts per K (best by likelihood).
    seed
        RNG seed.

    Returns
    -------
    dict
        ``{
            "best_k": int,
            "bic_per_k": [...k_max],
            "is_unimodal": bool,           # True if best_k == 1
            "best_model_means": [...best_k],
            "best_model_stds": [...best_k],
            "best_model_weights": [...best_k],
            "best_model_assignments": [...n_subjects],
        }``
    """
    r = np.asarray(residuals, dtype=np.float64).reshape(-1, 1)
    finite_mask = np.isfinite(r.ravel())
    if finite_mask.sum() < k_max:
        raise ValueError(
            f"Need at least k_max={k_max} finite residuals; got {int(finite_mask.sum())}"
        )
    r_fit = r[finite_mask]

    bics = []
    fitted = []
    for k in range(1, k_max + 1):
        gmm = GaussianMixture(
            n_components=k, n_init=n_init, random_state=seed, covariance_type="diag",
        )
        gmm.fit(r_fit)
        bics.append(float(gmm.bic(r_fit)))
        fitted.append(gmm)
    best_k_idx = int(np.argmin(bics))
    best = fitted[best_k_idx]
    best_k = best_k_idx + 1

    # Assign labels (via posterior argmax) for ALL subjects (including NaN as -1).
    assignments = np.full(r.size, -1, dtype=np.int64)
    if finite_mask.any():
        labels = best.predict(r_fit)
        assignments[finite_mask] = labels

    means = best.means_.ravel().tolist()
    stds = np.sqrt(best.covariances_.ravel()).tolist()
    weights = best.weights_.tolist()
    # Sort components by mean so labels are stable.
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
        "is_unimodal": bool(best_k == 1),
        "best_model_means": means_sorted,
        "best_model_stds": stds_sorted,
        "best_model_weights": weights_sorted,
        "best_model_assignments": assignments_sorted.tolist(),
        "config": {
            "k_max": int(k_max),
            "n_init": int(n_init),
            "seed": int(seed),
        },
    }
