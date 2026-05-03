"""Cell-type name list utilities.

Single source of truth for the "truncate-or-pad ``cell_type_names`` to
match the model's ``n_ct``" pattern that previously lived as a 2-line
in-line block in 9+ interpretability orchestrators (CC7 in
``docs/code_reviews/2026-05-02_full_review_C_FIXES.md``).

The pattern is:

    ct_names = list(CELL_TYPE_ORDER)[:n_ct]                       # truncate
    if len(ct_names) < n_ct:                                      # pad
        ct_names = ct_names + [f"ct_{c}" for c in range(len(ct_names), n_ct)]

Centralising it here means a future change to the placeholder convention
(e.g. ``unknown_ct_{c}``) only has to land in one place, and the
behaviour is unit-tested.
"""
from __future__ import annotations

from collections.abc import Iterable


def pad_cell_type_names(
    names: Iterable[str],
    n_ct: int,
    prefix: str = "ct_",
) -> list[str]:
    """Truncate or pad a cell-type-name list to length ``n_ct``.

    Parameters
    ----------
    names
        Source name list (typically ``CELL_TYPE_ORDER`` or the
        ``cell_type_names_used`` field from a sibling summary JSON).
    n_ct
        Target length, equal to the model's actual cell-type axis size.
    prefix
        Placeholder prefix for the pad-with-index branch. Default
        ``"ct_"`` matches all 9 historical call sites.

    Returns
    -------
    list[str]
        Length-``n_ct`` list. If ``len(names) >= n_ct``, returns
        ``names[:n_ct]``. Otherwise pads with
        ``[f"{prefix}{i}" for i in range(len(names), n_ct)]``.

    Raises
    ------
    ValueError
        If ``n_ct`` is negative.
    """
    if n_ct < 0:
        raise ValueError(f"n_ct must be non-negative, got {n_ct}")
    out = list(names)[:n_ct]
    if len(out) < n_ct:
        out = out + [f"{prefix}{i}" for i in range(len(out), n_ct)]
    return out
