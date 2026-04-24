"""Conditional mutual information of cell-type expression with resilience, given pathology.

The standard "does cell-type X carry signal beyond pathology Z?" question is:

    I(X; Y | Z)

where Y is the resilience composite (or its residual). We approximate via
the residualization trick, which is exact under linear-Gaussian assumptions
for ``regressor="linear"``; for nonlinear Z→X dependencies, set
``regressor="rf"`` to use a Random Forest residualizer (still
approximate but distribution-free).

Procedure:
  1. Fit Z → X regressor (linear or RF).
  2. Compute residuals X_resid = X − ẑ.
  3. Estimate I(X_resid; Y) via the KSG (Kraskov–Stögbauer–Grassberger)
     estimator from ``sklearn.feature_selection.mutual_info_regression``.

Inputs to ``conditional_mi_per_celltype`` can be EITHER:
  - per-CT scalar (per-subject mean-across-genes for that cell type), or
  - per-CT vector of length n_genes (the full per-CT pseudobulk).

The vector path uses sklearn's multivariate KSG (passes all gene features
into ``mutual_info_regression`` and reports the MAX MI across genes per CT,
since per-feature MI is what KSG actually returns).
"""
from __future__ import annotations

from typing import Literal, Sequence

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LinearRegression


def _residualize(
    X: np.ndarray, Z: np.ndarray, regressor: Literal["linear", "rf"], seed: int,
) -> np.ndarray:
    """Return X minus the regressor's prediction from Z. Per-column independent."""
    if regressor == "linear":
        lr = LinearRegression()
        lr.fit(Z, X)
        return X - lr.predict(Z)
    if regressor == "rf":
        rf = RandomForestRegressor(
            n_estimators=200, max_depth=None, n_jobs=-1, random_state=seed,
        )
        rf.fit(Z, X)
        return X - rf.predict(Z)
    raise ValueError(f"Unknown regressor: {regressor!r}")


def conditional_mi_per_celltype(
    expression_per_subject: np.ndarray,
    resilience_y: np.ndarray,
    pathology_z: np.ndarray,
    *,
    cell_type_names: Sequence[str] | None = None,
    seed: int = 42,
    n_neighbors: int = 5,
    regressor: Literal["linear", "rf"] = "linear",
    min_samples: int = 30,
    aggregation: Literal["mean", "max", "vector"] = "max",
) -> dict:
    """Per-cell-type conditional MI: I(CT_expression; Y | pathology Z).

    Parameters
    ----------
    expression_per_subject
        Either ``(n_subjects, n_celltypes)`` (scalar per CT, equivalent to
        ``aggregation="mean"`` already applied) OR ``(n_subjects, n_celltypes,
        n_genes)`` (full per-CT gene vector, recommended for the honest
        "does CT carry signal" claim).
    resilience_y
        Shape ``(n_subjects,)`` — the resilience composite or its residual.
    pathology_z
        Shape ``(n_subjects, n_pathology_features)`` — covariates to condition on.
    cell_type_names
        Optional names; default ``CT_<i>``.
    seed
        RNG seed.
    n_neighbors
        KSG neighborhood (default 5).
    regressor
        ``"linear"`` (default; fast, exact under linear-Gaussian) or ``"rf"``
        (slower; captures nonlinear Z→X).
    min_samples
        Minimum finite samples per CT to compute MI; below this returns NaN.
    aggregation
        For 3D ``expression_per_subject``: how to reduce per-CT gene-vector
        MI to a single number per CT.
        - ``"max"``: max MI across genes (best per-CT signal). Default.
        - ``"mean"``: mean MI across genes.
        - ``"vector"``: keep per-gene MI in a separate ``per_gene_mi`` field.

    Returns
    -------
    dict
        ``{
            "per_cell_type": [
                {
                    "cell_type": str,
                    "unconditional_mi": float,
                    "conditional_mi_given_pathology": float,
                    "delta": float,  # how much MI is "explained away" by pathology
                    "n_used": int,
                    ... (per_gene_mi if aggregation="vector")
                },
                ...
            ],
            "config": {...}
        }``
    """
    arr = np.asarray(expression_per_subject)
    is_vector_input = arr.ndim == 3
    if not is_vector_input:
        if arr.ndim != 2:
            raise ValueError(f"expression must be 2D or 3D; got shape {arr.shape}")
    n_subj = arr.shape[0]
    n_ct = arr.shape[1]
    Y = np.asarray(resilience_y, dtype=np.float64).ravel()
    Z = np.asarray(pathology_z, dtype=np.float64)
    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)
    if cell_type_names is None:
        cell_type_names = [f"CT_{i}" for i in range(n_ct)]

    rng_state = int(seed)
    per_ct = []
    for ct in range(n_ct):
        if is_vector_input:
            x_full = arr[:, ct, :].astype(np.float64)
            mask = (
                np.isfinite(x_full).all(axis=1)
                & np.isfinite(Y)
                & np.isfinite(Z).all(axis=1)
            )
        else:
            x_full = arr[:, ct].astype(np.float64).reshape(-1, 1)
            mask = (
                np.isfinite(x_full).all(axis=1)
                & np.isfinite(Y)
                & np.isfinite(Z).all(axis=1)
            )

        if mask.sum() < min_samples:
            per_ct.append({
                "cell_type": str(cell_type_names[ct]),
                "unconditional_mi": float("nan"),
                "conditional_mi_given_pathology": float("nan"),
                "delta": float("nan"),
                "n_used": int(mask.sum()),
                "note": f"insufficient finite samples (n<{min_samples})",
            })
            continue

        xs, ys, zs = x_full[mask], Y[mask], Z[mask]
        # Unconditional MI per feature → aggregate to scalar.
        mi_unc_per_feat = mutual_info_regression(
            xs, ys, n_neighbors=n_neighbors, random_state=rng_state,
        )
        # Conditional MI: residualize each X feature, then MI per feature.
        x_resid = _residualize(xs, zs, regressor, seed=rng_state)
        if x_resid.ndim == 1:
            x_resid = x_resid.reshape(-1, 1)
        mi_cond_per_feat = mutual_info_regression(
            x_resid, ys, n_neighbors=n_neighbors, random_state=rng_state,
        )
        if aggregation == "max":
            mi_unc = float(np.max(mi_unc_per_feat))
            mi_cond = float(np.max(mi_cond_per_feat))
        elif aggregation == "mean":
            mi_unc = float(np.mean(mi_unc_per_feat))
            mi_cond = float(np.mean(mi_cond_per_feat))
        elif aggregation == "vector":
            mi_unc = float(np.max(mi_unc_per_feat))
            mi_cond = float(np.max(mi_cond_per_feat))
        else:
            raise ValueError(f"Unknown aggregation: {aggregation!r}")

        entry = {
            "cell_type": str(cell_type_names[ct]),
            "unconditional_mi": mi_unc,
            "conditional_mi_given_pathology": mi_cond,
            "delta": mi_unc - mi_cond,
            "n_used": int(mask.sum()),
        }
        if aggregation == "vector":
            entry["per_gene_unconditional_mi"] = mi_unc_per_feat.tolist()
            entry["per_gene_conditional_mi"] = mi_cond_per_feat.tolist()
        per_ct.append(entry)

    return {
        "per_cell_type": per_ct,
        "config": {
            "n_subjects_total": int(n_subj),
            "n_pathology_features": int(Z.shape[1]),
            "n_neighbors": int(n_neighbors),
            "seed": int(seed),
            "regressor": str(regressor),
            "min_samples": int(min_samples),
            "aggregation": str(aggregation),
            "input_was_vector": bool(is_vector_input),
        },
    }
