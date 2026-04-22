"""Variance decomposition for the ResDec-H3 composite prediction.

The composite prediction is y_hat = y_tabpfn + f_1, where y_tabpfn is the
outer-fold TabPFN-2.6 scalar prediction and f_1 is the neural head's
residual contribution. The identity

    Var(y) = Var(y_tabpfn) + Var(f_1) + 2 * Cov(y_tabpfn, f_1) + Var(resid)

follows from

    y = y_tabpfn + f_1 + resid,         resid = y - (y_tabpfn + f_1)

and Var(A + B + R) expanded with the cross-covariance structure. Because
the neural head is trained to produce a scalar that fuses with the frozen
TabPFN prediction, `f_1` and `y_tabpfn` are NOT assumed independent —
the cross term Cov(y_tabpfn, f_1) quantifies how much of the improvement
over TabPFN comes from alignment vs. orthogonal signal.

This module is pure: inputs are NumPy arrays, output is a nested dict,
no I/O. Orchestration (loading per-fold npz, metadata join, JSON write)
lives in scripts/redesign/interpretability/variance_decomposition.py.

All variances and covariances use the unbiased sample estimator
(``ddof=1``) so the decomposition reports finite-sample quantities
without a systematic bias.
"""
from __future__ import annotations

import logging
from typing import Mapping

import numpy as np

logger = logging.getLogger(__name__)


def _component_stats(
    y_true: np.ndarray,
    y_tabpfn: np.ndarray,
    f1_residual: np.ndarray,
) -> dict[str, float]:
    """Compute the five variance components + explained fraction on a single group.

    Returns a dict with keys
        ``var_y``, ``var_tabpfn``, ``var_f1``, ``cov_tabpfn_f1``,
        ``var_resid``, ``total_explained_fraction``, ``n``.

    When ``n < 2`` the sample variance is undefined; all variance / covariance
    fields are returned as ``float('nan')`` and ``total_explained_fraction``
    is likewise NaN. This matches the convention used by
    :func:`numpy.var` with ``ddof=1`` (which emits a warning and returns NaN).
    """
    n = int(y_true.shape[0])
    resid = y_true - (y_tabpfn + f1_residual)

    if n < 2:
        return {
            "var_y": float("nan"),
            "var_tabpfn": float("nan"),
            "var_f1": float("nan"),
            "cov_tabpfn_f1": float("nan"),
            "var_resid": float("nan"),
            "total_explained_fraction": float("nan"),
            "n": n,
        }

    var_y = float(np.var(y_true, ddof=1))
    var_tabpfn = float(np.var(y_tabpfn, ddof=1))
    var_f1 = float(np.var(f1_residual, ddof=1))
    var_resid = float(np.var(resid, ddof=1))
    # np.cov returns the 2x2 covariance matrix; off-diagonal is the scalar cov.
    cov_tabpfn_f1 = float(np.cov(y_tabpfn, f1_residual, ddof=1)[0, 1])

    total_explained_fraction = (
        1.0 - var_resid / var_y if var_y > 0.0 else float("nan")
    )

    return {
        "var_y": var_y,
        "var_tabpfn": var_tabpfn,
        "var_f1": var_f1,
        "cov_tabpfn_f1": cov_tabpfn_f1,
        "var_resid": var_resid,
        "total_explained_fraction": float(total_explained_fraction),
        "n": n,
    }


def decompose_variance(
    y_true: np.ndarray,
    y_tabpfn: np.ndarray,
    f1_residual: np.ndarray,
    *,
    subgroups: Mapping[str, np.ndarray] | None = None,
) -> dict:
    """Decompose Var(y_true) into TabPFN, neural residual, cross, and residual-error terms.

    Parameters
    ----------
    y_true, y_tabpfn, f1_residual : np.ndarray, shape ``[N]``
        Per-subject vectors. ``y_tabpfn + f1_residual`` is the composite
        prediction; ``resid = y_true - (y_tabpfn + f1_residual)`` is the
        residual error used in the decomposition.
    subgroups : dict[str, np.ndarray] | None
        Optional mapping ``{"by_xyz": labels[N]}`` — each entry triggers a
        split by unique label value. Entries with ``None`` / NaN label are
        excluded. The key is passed through verbatim to the output dict.

    Returns
    -------
    dict
        ::

            {
              "global": {var_y, var_tabpfn, var_f1, cov_tabpfn_f1,
                         var_resid, total_explained_fraction, n},
              "<subgroup_key>": {
                  "<label_1>": {... same schema as global ...},
                  "<label_2>": {...},
                  ...
              },
              ...
            }

        Additivity always holds:
        ``var_y == var_tabpfn + var_f1 + 2 * cov_tabpfn_f1 + var_resid``
        (up to floating-point precision).

    Notes
    -----
    Subgroup labels are sorted lexicographically as strings before
    iterating, so e.g. numeric-string labels ``"0"``, ``"1"``, ``"10"``,
    ``"2"`` sort as ``["0", "1", "10", "2"]``. Callers wanting natural
    numeric ordering should pre-process labels to a zero-padded or
    otherwise lex-sortable form.
    """
    y_true = np.asarray(y_true)
    y_tabpfn = np.asarray(y_tabpfn)
    f1_residual = np.asarray(f1_residual)

    if y_true.shape != y_tabpfn.shape or y_true.shape != f1_residual.shape:
        raise ValueError(
            "y_true, y_tabpfn, f1_residual must share shape; got "
            f"{y_true.shape}, {y_tabpfn.shape}, {f1_residual.shape}."
        )
    if y_true.ndim != 1:
        raise ValueError(f"Inputs must be 1-D; got ndim={y_true.ndim}.")

    n = int(y_true.shape[0])

    out: dict = {"global": _component_stats(y_true, y_tabpfn, f1_residual)}

    if subgroups:
        for key, labels in subgroups.items():
            labels = np.asarray(labels, dtype=object)
            if labels.shape[0] != n:
                raise ValueError(
                    f"Subgroup '{key}' has length {labels.shape[0]}; "
                    f"expected {n}."
                )
            # Exclude None / NaN entries; pandas-style "missing" handling.
            valid_mask = np.array(
                [not (lbl is None or (isinstance(lbl, float) and np.isnan(lbl)))
                 for lbl in labels],
                dtype=bool,
            )
            valid_labels = labels[valid_mask]
            unique_labels = sorted({lbl for lbl in valid_labels.tolist()},
                                   key=lambda x: str(x))

            if not unique_labels:
                logger.warning(
                    "Subgroup %r has 0 non-null labels; emitting empty dict "
                    "for this subgroup.",
                    key,
                )
                out[key] = {}
                continue

            group_out: dict = {}
            for lbl in unique_labels:
                grp_mask = (labels == lbl) & valid_mask
                group_out[str(lbl)] = _component_stats(
                    y_true[grp_mask],
                    y_tabpfn[grp_mask],
                    f1_residual[grp_mask],
                )
            out[key] = group_out

    return out


__all__ = ["decompose_variance"]
