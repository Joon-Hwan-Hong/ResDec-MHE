"""Conditional mutual information of cell-type expression with resilience, given pathology.

The standard "does cell-type X carry signal beyond pathology Z?" question is:

    I(X; Y | Z)

where Y is the resilience composite (or its residual). We approximate this
via the residualization trick, which is exact under linear-Gaussian
assumptions:

    1. Fit linear regression Z → X (per cell-type expression vector or per
       cell-type mean expression scalar).
    2. Compute residuals X_resid = X − ẑ.
    3. Estimate I(X_resid; Y) via the KSG (Kraskov–Stögbauer–Grassberger)
       estimator implemented in scikit-learn's
       ``feature_selection.mutual_info_regression``.

The result quantifies how much expression carries about resilience that is
not already explained by the pathology covariates.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LinearRegression


def conditional_mi_per_celltype(
    expression_per_subject: np.ndarray,
    resilience_y: np.ndarray,
    pathology_z: np.ndarray,
    *,
    cell_type_names: Sequence[str] | None = None,
    seed: int = 42,
    n_neighbors: int = 5,
) -> dict:
    """Per-cell-type conditional MI: I(CT_mean_expression; Y | pathology Z).

    Parameters
    ----------
    expression_per_subject
        Shape ``(n_subjects, n_celltypes)`` — per-subject mean expression
        per cell type (already aggregated across genes; e.g., the per-CT
        gene-mean from the precomputed pseudobulk).
    resilience_y
        Shape ``(n_subjects,)`` — the resilience composite or its residual.
    pathology_z
        Shape ``(n_subjects, n_pathology_features)`` — pathology covariates
        (gpath, amyloid, tangles, etc.) to condition on.
    cell_type_names
        Optional names for cell types.
    seed
        RNG seed for the KSG estimator (used for jitter ties).
    n_neighbors
        KSG neighborhood (default 5; larger = lower variance, more bias).

    Returns
    -------
    dict
        ``{
            "per_cell_type": [
                {
                    "cell_type": str,
                    "unconditional_mi": float,
                    "conditional_mi_given_pathology": float,
                    "delta": float,  # unconditional - conditional;
                                     # how much MI is "explained away" by pathology
                },
                ...
            ],
            "config": {...}
        }``
    """
    n_subj, n_ct = expression_per_subject.shape
    Y = np.asarray(resilience_y, dtype=np.float64).ravel()
    Z = np.asarray(pathology_z, dtype=np.float64)
    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)
    if cell_type_names is None:
        cell_type_names = [f"CT_{i}" for i in range(n_ct)]

    rng_state = int(seed)
    per_ct = []
    for ct in range(n_ct):
        x = expression_per_subject[:, ct].astype(np.float64).reshape(-1, 1)
        # Drop subjects with NaN in any of (x, y, z).
        mask = (
            np.isfinite(x).all(axis=1)
            & np.isfinite(Y)
            & np.isfinite(Z).all(axis=1)
        )
        if mask.sum() < 30:
            per_ct.append({
                "cell_type": str(cell_type_names[ct]),
                "unconditional_mi": float("nan"),
                "conditional_mi_given_pathology": float("nan"),
                "delta": float("nan"),
                "n_used": int(mask.sum()),
                "note": "insufficient finite samples (n<30)",
            })
            continue
        xs, ys, zs = x[mask], Y[mask], Z[mask]
        # Unconditional MI.
        mi_unc = float(mutual_info_regression(
            xs, ys, n_neighbors=n_neighbors, random_state=rng_state,
        )[0])
        # Conditional MI via residualization: regress Z out of X, then I(X_resid; Y).
        lr = LinearRegression()
        lr.fit(zs, xs.ravel())
        x_resid = xs.ravel() - lr.predict(zs)
        mi_cond = float(mutual_info_regression(
            x_resid.reshape(-1, 1), ys, n_neighbors=n_neighbors, random_state=rng_state,
        )[0])
        per_ct.append({
            "cell_type": str(cell_type_names[ct]),
            "unconditional_mi": mi_unc,
            "conditional_mi_given_pathology": mi_cond,
            "delta": mi_unc - mi_cond,
            "n_used": int(mask.sum()),
        })

    return {
        "per_cell_type": per_ct,
        "config": {
            "n_subjects_total": int(n_subj),
            "n_pathology_features": int(Z.shape[1]),
            "n_neighbors": int(n_neighbors),
            "seed": int(seed),
        },
    }
