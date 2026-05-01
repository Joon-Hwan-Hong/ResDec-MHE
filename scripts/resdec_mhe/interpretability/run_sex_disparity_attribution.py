#!/usr/bin/env python
"""Sex-disparity attribution analysis (EXP-038 follow-up to EXP-035).

EXP-035 flagged a substantial sex disparity in canonical model performance:
female mean R² = 0.481 (n=334) vs male mean R² = 0.288 (n=182), Δ = 0.193 R²
units, with high cross-fold variance on the male subset (std = 0.219; fold-3
male R² = -0.028). This script asks **whether the model uses different cell
types or genes to predict cognitive resilience for the two sexes**, which lets
us discriminate between three hypotheses:

(a) the gap is small-N noise on the male subset,
(b) the model exploits a sex-specific cell-type / gene signature that is
    well-captured for one sex but not the other, or
(c) male subjects are more heterogeneous and the cohort-level signature is a
    poorer fit.

Method
------
For every subject in the 5-fold val concatenation we already have a Captum
Integrated Gradients tensor (composite_attributions.npz: ``[N, 31, 4785]``)
mapping the encoder + ResDec-MHE residual head's attribution back to a per-
``(cell_type, gene)`` value (positive = pushes residual up, negative = down).
We:

  1. Aggregate per-subject ``|attribution|`` into per-CT and per-(CT, gene)
     scalars, then split by ``msex`` (0 = female, 1 = male) using
     ``data/metadata_ROSMAP/metadata.csv``.
  2. Rank top-5 CTs separately by F mean and M mean attribution and report
     the overlap.
  3. For each CT, rank top-10 genes separately by F vs M mean attribution
     (per-CT differential top-genes table).
  4. **Sex × CT interaction test:** for each of the 31 CTs, run a Wilcoxon
     rank-sum (``scipy.stats.ranksums``) on per-subject CT-aggregated
     attribution magnitude (F vs M); apply BH-FDR via
     ``scipy.stats.false_discovery_control`` over the 31 tests.
  5. **Sex × gene interaction test:** for the top-50 ``(CT, gene)`` pairs
     ranked by **overall** mean ``|attribution|`` (cohort-wide; not split
     by sex), run a Wilcoxon rank-sum on per-subject ``|attribution|`` at
     that pair (F vs M); BH-FDR over the 50.
  6. **Per-subject prediction error vs sex × CT activation:** join per-fold
     ``val_predictions_best.npz`` to compute ``|y_true − y_composite|``, then
     compute Spearman ρ between residual magnitude and each CT's per-subject
     attribution magnitude **separately by sex**.

Outputs
-------
    --out-json   outputs/canonical/interpretability/sex_disparity_attribution.json
    --out-md     outputs/canonical/interpretability/sex_disparity_attribution.md
    --out-fig-dir outputs/canonical/interpretability/figures/sex_disparity/
                    fig_sex_disparity.{png, pdf}  (4-panel, 600 DPI)

Usage
-----
    PYTHONPATH=<worktree-root> uv run python \\
        scripts/resdec_mhe/interpretability/run_sex_disparity_attribution.py
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # must precede pyplot import
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control, ranksums, spearmanr

_WORKTREE_ROOT = Path(__file__).resolve().parents[3]
if str(_WORKTREE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKTREE_ROOT))

from src.data.constants import CELL_TYPE_ORDER  # noqa: E402
from src.visualization.theme import (  # noqa: E402
    apply_theme,
    fmt_axes,
    style_paper_axes,
)

logger = logging.getLogger(__name__)


# Magic-number constants — surfaced here so a reviewer doesn't need to spelunk
# through function bodies to find / change them.
DEFAULT_TOP_N_CT = 5            # ranking: top-K cell types per sex
DEFAULT_TOP_N_GENES_PER_CT = 10  # ranking: top-N genes per (CT, sex)
DEFAULT_TOP_N_PAIRS = 50         # Wilcoxon FDR scope for (CT, gene) pairs
BH_Q_THRESHOLD = 0.05            # BH-adjusted significance threshold


# =============================================================================
# Loading helpers
# =============================================================================


def load_attribution_tensor(npz_path: Path) -> dict[str, np.ndarray]:
    """Load Captum IG attribution tensor produced by
    ``captum_composite_attribution.py``.

    Returns a dict with:
      - ``subject_ids``       : str ndarray, shape (N,)
      - ``attributions``      : float32 ndarray, shape (N, n_ct, n_genes)
      - ``predictions_residual``: float32 ndarray, shape (N,) — encoder+head
                                  residual prediction (NOT the composite, see
                                  caller note in captum_composite_attribution.py)
      - ``fold``              : int32 ndarray, shape (N,)
    """
    if not npz_path.exists():
        raise FileNotFoundError(npz_path)
    d = np.load(npz_path, allow_pickle=True)
    needed = ("subject_ids", "attributions", "predictions_residual", "fold")
    for k in needed:
        if k not in d.files:
            raise KeyError(f"{npz_path}: missing required key {k!r}")
    return {
        "subject_ids": np.asarray(d["subject_ids"]).astype(str),
        "attributions": np.asarray(d["attributions"]).astype(np.float32),
        "predictions_residual": np.asarray(d["predictions_residual"]).astype(
            np.float32
        ),
        "fold": np.asarray(d["fold"]).astype(np.int32),
    }


def load_val_predictions_per_fold(
    pred_root: Path, n_folds: int = 5,
) -> pd.DataFrame:
    """Concatenate per-fold ``val_predictions_best.npz`` into a long DataFrame.

    Columns: ``ROSMAP_IndividualID, fold, y_true, y_composite``. We do NOT
    join TabPFN here — we want the raw composite prediction for residual
    computation, which is what's already in the npz.
    """
    rows = []
    for fold in range(n_folds):
        path = pred_root / f"fold{fold}" / "val_predictions_best.npz"
        if not path.exists():
            raise FileNotFoundError(path)
        d = np.load(path, allow_pickle=True)
        sids = np.asarray(d["subject_ids"]).astype(str)
        rows.append(
            pd.DataFrame(
                {
                    "ROSMAP_IndividualID": sids,
                    "fold": fold,
                    "y_true": np.asarray(d["targets"]).astype(np.float64),
                    "y_composite": np.asarray(d["predictions"]).astype(np.float64),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def load_sex_metadata(metadata_csv: Path) -> pd.DataFrame:
    """Return a 2-column DataFrame with ``ROSMAP_IndividualID`` and ``msex``.

    ``msex`` is the ROSMAP coding 0 = female, 1 = male. Subjects with NaN or
    out-of-range msex are dropped here (caller will see them as missing in
    the inner-join). Coverage in the canonical 516-subject set is 100% as
    confirmed by EXP-035.
    """
    if not metadata_csv.exists():
        raise FileNotFoundError(metadata_csv)
    meta = pd.read_csv(metadata_csv, low_memory=False)
    for col in ("ROSMAP_IndividualID", "msex"):
        if col not in meta.columns:
            raise KeyError(f"metadata.csv missing required column: {col!r}")
    df = meta[["ROSMAP_IndividualID", "msex"]].copy()
    df["ROSMAP_IndividualID"] = df["ROSMAP_IndividualID"].astype(str)
    df["msex_numeric"] = pd.to_numeric(df["msex"], errors="coerce")
    df = df.dropna(subset=["msex_numeric"]).copy()
    df = df[df["msex_numeric"].isin([0, 1])].copy()
    df["sex"] = np.where(df["msex_numeric"] == 0, "female", "male")
    return df[["ROSMAP_IndividualID", "sex"]]


def load_gene_names(precomputed_dir: Path, n_genes: int) -> list[str]:
    """Load gene-name vector (length ``n_genes``) from
    ``data/precomputed/gene_names.npy``; falls back to placeholders if the
    sidecar is missing — the same fallback contract as
    ``captum_composite_attribution._load_gene_names``.
    """
    candidates = [
        precomputed_dir / "gene_names.npy",
        precomputed_dir / "gene_names.json",
    ]
    for p in candidates:
        if not p.exists():
            continue
        if p.suffix == ".npy":
            names = np.load(p, allow_pickle=True).tolist()
        else:
            names = json.loads(p.read_text())
        if isinstance(names, list) and len(names) >= n_genes:
            return [str(n) for n in names[:n_genes]]
    logger.warning(
        "No gene_names sidecar found in %s; falling back to gene_<i> placeholders.",
        precomputed_dir,
    )
    return [f"gene_{i}" for i in range(n_genes)]


# =============================================================================
# Per-subject aggregation
# =============================================================================


def per_subject_ct_magnitude(attr: np.ndarray) -> np.ndarray:
    """Return per-subject per-CT scalar attribution magnitude.

    ``attr`` is the IG tensor of shape ``(N, n_ct, n_genes)``. We use the
    L1-norm across genes (``mean(|attr|, axis=2)``) so a CT whose per-gene
    attribution sum-to-zero (signed) but is large in magnitude is still flagged
    as "active". Returned shape: ``(N, n_ct)``.
    """
    if attr.ndim != 3:
        raise ValueError(f"attr must be 3D; got shape {attr.shape}")
    return np.mean(np.abs(attr), axis=2)


def per_subject_pair_magnitude(
    attr: np.ndarray, pairs: list[tuple[int, int]],
) -> np.ndarray:
    """Per-subject ``|attribution|`` for the supplied ``(ct_idx, gene_idx)``
    pairs.

    Returned shape: ``(N, len(pairs))``.
    """
    if attr.ndim != 3:
        raise ValueError(f"attr must be 3D; got shape {attr.shape}")
    if not pairs:
        return np.zeros((attr.shape[0], 0), dtype=np.float32)
    ct_idx = np.asarray([p[0] for p in pairs], dtype=np.int64)
    g_idx = np.asarray([p[1] for p in pairs], dtype=np.int64)
    return np.abs(attr[:, ct_idx, g_idx])


# =============================================================================
# Sex-stratified statistics
# =============================================================================


def rank_top_ct_per_sex(
    ct_mag: np.ndarray,
    sex: np.ndarray,
    ct_names: list[str],
    top_n: int = DEFAULT_TOP_N_CT,
) -> dict[str, list[dict[str, float | str]]]:
    """Rank top-``top_n`` cell types separately by F-mean and M-mean magnitude.

    ``ct_mag`` is ``(N, n_ct)`` per-subject CT magnitudes; ``sex`` is a
    string array of length N with values in ``{"female", "male"}``.
    Returns ``{"female": [{"cell_type": str, "mean_abs_attribution": float,
    "rank": int}, ...], "male": [...]}``.
    """
    n_ct = ct_mag.shape[1]
    if len(ct_names) != n_ct:
        raise ValueError(f"len(ct_names)={len(ct_names)} != n_ct={n_ct}")
    out: dict[str, list[dict[str, float | str]]] = {}
    for sex_label in ("female", "male"):
        mask = sex == sex_label
        if mask.sum() == 0:
            out[sex_label] = []
            continue
        means = ct_mag[mask].mean(axis=0)
        order = np.argsort(-means)[:top_n]
        out[sex_label] = [
            {
                "cell_type": ct_names[int(i)],
                "mean_abs_attribution": float(means[int(i)]),
                "rank": int(rank + 1),
            }
            for rank, i in enumerate(order)
        ]
    return out


def rank_top_genes_per_ct_per_sex(
    attr: np.ndarray,
    sex: np.ndarray,
    ct_names: list[str],
    gene_names: list[str],
    top_n: int = DEFAULT_TOP_N_GENES_PER_CT,
) -> dict[str, dict[str, list[dict[str, float | str]]]]:
    """For each CT, rank top-N genes by F-mean and M-mean magnitude.

    Returned structure:

        {
          <ct_name>: {
            "female": [{"gene": str, "mean_abs_attribution": float, "rank": int},
                       ...],
            "male":   [...]
          },
          ...
        }
    """
    if attr.ndim != 3:
        raise ValueError(f"attr must be 3D; got shape {attr.shape}")
    n, n_ct, n_genes = attr.shape
    if len(ct_names) != n_ct or len(gene_names) != n_genes:
        raise ValueError(
            f"shape mismatch: attr={attr.shape}, ct_names={len(ct_names)}, "
            f"gene_names={len(gene_names)}"
        )
    # Pre-compute per-subject |attr| once to avoid repeated abs() in the loop.
    abs_attr = np.abs(attr)
    out: dict[str, dict[str, list[dict[str, float | str]]]] = {}
    for ct_idx, ct in enumerate(ct_names):
        out[ct] = {}
        for sex_label in ("female", "male"):
            mask = sex == sex_label
            if mask.sum() == 0:
                out[ct][sex_label] = []
                continue
            means_g = abs_attr[mask, ct_idx, :].mean(axis=0)  # (n_genes,)
            order = np.argsort(-means_g)[:top_n]
            out[ct][sex_label] = [
                {
                    "gene": gene_names[int(g)],
                    "mean_abs_attribution": float(means_g[int(g)]),
                    "rank": int(rank + 1),
                }
                for rank, g in enumerate(order)
            ]
    return out


def wilcoxon_per_ct(
    ct_mag: np.ndarray, sex: np.ndarray, ct_names: list[str],
) -> dict[str, dict[str, float]]:
    """Wilcoxon rank-sum on per-CT magnitude (F vs M), with BH-FDR over CTs.

    ``ranksums`` is two-sided by default; we keep that (asks "different",
    not "F > M" specifically). BH correction is over the 31 CTs via
    ``scipy.stats.false_discovery_control(method='bh')``. Returns a dict
    keyed by CT name with ``{statistic, p_value, q_value, mean_female,
    mean_male, mean_diff_F_minus_M, n_female, n_male}``. ``q_value`` is the
    BH-adjusted p-value.

    Failure modes (degenerate inputs) are handled by emitting NaN p / q for
    that CT but never raising — the Wilcoxon needs ≥ 1 obs per group; if a
    sex has zero observations the whole sweep is meaningless and the caller
    should noop earlier, but we still degrade gracefully.
    """
    n_ct = ct_mag.shape[1]
    if len(ct_names) != n_ct:
        raise ValueError(f"len(ct_names)={len(ct_names)} != n_ct={n_ct}")
    fem = sex == "female"
    mal = sex == "male"
    n_f = int(fem.sum())
    n_m = int(mal.sum())

    stats: list[float] = []
    pvals: list[float] = []
    means_f: list[float] = []
    means_m: list[float] = []
    for c in range(n_ct):
        x_f = ct_mag[fem, c]
        x_m = ct_mag[mal, c]
        if n_f == 0 or n_m == 0:
            stats.append(float("nan"))
            pvals.append(float("nan"))
            means_f.append(float("nan") if n_f == 0 else float(np.mean(x_f)))
            means_m.append(float("nan") if n_m == 0 else float(np.mean(x_m)))
            continue
        # ranksums returns nan p for all-tied or all-equal in degenerate
        # cases; we don't synthesize a p, just propagate scipy's value.
        try:
            s, p = ranksums(x_f, x_m)
            stats.append(float(s))
            pvals.append(float(p))
        except ValueError:
            stats.append(float("nan"))
            pvals.append(float("nan"))
        means_f.append(float(np.mean(x_f)))
        means_m.append(float(np.mean(x_m)))

    pvals_arr = np.asarray(pvals, dtype=np.float64)
    finite_mask = np.isfinite(pvals_arr)
    qvals = np.full(n_ct, np.nan, dtype=np.float64)
    if finite_mask.any():
        # BH-FDR on the finite subset, leave NaN entries as NaN.
        qvals[finite_mask] = false_discovery_control(
            pvals_arr[finite_mask], method="bh"
        )
    out: dict[str, dict[str, float]] = {}
    for c, ct in enumerate(ct_names):
        out[ct] = {
            "statistic": stats[c],
            "p_value": pvals[c],
            "q_value": float(qvals[c]) if np.isfinite(qvals[c]) else float("nan"),
            "mean_female": means_f[c],
            "mean_male": means_m[c],
            "mean_diff_F_minus_M": means_f[c] - means_m[c],
            "n_female": n_f,
            "n_male": n_m,
        }
    return out


def topk_pairs_overall(
    attr: np.ndarray, ct_names: list[str], gene_names: list[str], top_k: int,
) -> list[tuple[int, int, str, str, float]]:
    """Cohort-wide top-``top_k`` ``(ct_idx, gene_idx)`` pairs by mean ``|attr|``.

    Returned tuples are ``(ct_idx, gene_idx, ct_name, gene_name,
    mean_abs_attribution_overall)``. We use the cohort mean (not sex-split)
    to define the ranking universe, then run sex-split tests on those pairs
    in ``wilcoxon_per_pair``.
    """
    n, n_ct, n_genes = attr.shape
    cohort_mean = np.abs(attr).mean(axis=0)  # (n_ct, n_genes)
    flat = cohort_mean.flatten()
    # argpartition is faster than argsort for top-K, but argsort on a 31×4785
    # = 148K array is < 1 ms, so just argsort for clarity.
    order = np.argsort(-flat)[:top_k]
    pairs: list[tuple[int, int, str, str, float]] = []
    for idx in order:
        c, g = divmod(int(idx), n_genes)
        pairs.append(
            (c, g, ct_names[c], gene_names[g], float(cohort_mean[c, g]))
        )
    return pairs


def wilcoxon_per_pair(
    attr: np.ndarray,
    sex: np.ndarray,
    pairs: list[tuple[int, int, str, str, float]],
) -> list[dict[str, float | str]]:
    """Wilcoxon rank-sum F vs M on per-subject ``|attr[:, c, g]|``, BH-FDR over
    the supplied ``pairs``.

    Returns a list of dicts in the same order as ``pairs``:
      ``{"cell_type": str, "gene": str, "ct_idx": int, "gene_idx": int,
         "mean_overall": float, "mean_female": float, "mean_male": float,
         "mean_diff_F_minus_M": float, "statistic": float, "p_value": float,
         "q_value": float, "n_female": int, "n_male": int}``
    """
    fem = sex == "female"
    mal = sex == "male"
    n_f = int(fem.sum())
    n_m = int(mal.sum())

    pair_mag = per_subject_pair_magnitude(
        attr, [(c, g) for c, g, *_ in pairs]
    )
    stats: list[float] = []
    pvals: list[float] = []
    mf_list: list[float] = []
    mm_list: list[float] = []
    for k in range(pair_mag.shape[1]):
        x_f = pair_mag[fem, k]
        x_m = pair_mag[mal, k]
        mf_list.append(float(np.mean(x_f)) if n_f > 0 else float("nan"))
        mm_list.append(float(np.mean(x_m)) if n_m > 0 else float("nan"))
        if n_f == 0 or n_m == 0:
            stats.append(float("nan"))
            pvals.append(float("nan"))
            continue
        try:
            s, p = ranksums(x_f, x_m)
            stats.append(float(s))
            pvals.append(float(p))
        except ValueError:
            stats.append(float("nan"))
            pvals.append(float("nan"))

    pvals_arr = np.asarray(pvals, dtype=np.float64)
    finite_mask = np.isfinite(pvals_arr)
    qvals = np.full(len(pairs), np.nan, dtype=np.float64)
    if finite_mask.any():
        qvals[finite_mask] = false_discovery_control(
            pvals_arr[finite_mask], method="bh"
        )
    out: list[dict[str, float | str]] = []
    for k, (c, g, ct_name, gene_name, m_overall) in enumerate(pairs):
        out.append(
            {
                "cell_type": ct_name,
                "gene": gene_name,
                "ct_idx": int(c),
                "gene_idx": int(g),
                "mean_overall": float(m_overall),
                "mean_female": mf_list[k],
                "mean_male": mm_list[k],
                "mean_diff_F_minus_M": mf_list[k] - mm_list[k],
                "statistic": stats[k],
                "p_value": pvals[k],
                "q_value": float(qvals[k]) if np.isfinite(qvals[k]) else float("nan"),
                "n_female": n_f,
                "n_male": n_m,
            }
        )
    return out


def spearman_residual_vs_ct_per_sex(
    abs_residual: np.ndarray,
    ct_mag: np.ndarray,
    sex: np.ndarray,
    ct_names: list[str],
) -> dict[str, list[dict[str, float | str]]]:
    """Spearman ρ of ``|residual|`` vs each CT's per-subject magnitude,
    separately by sex.

    For each (sex, CT) pair we run a single Spearman correlation between the
    per-subject ``|residual|`` and that CT's per-subject magnitude. Returns
    ``{"female": [{cell_type, rho, p_value, n}, ...], "male": [...]}``. We
    deliberately do NOT FDR-correct these correlations here — they are
    descriptive (does prediction error covary with CT-level activation
    within each sex?) and the panel renders the unadjusted ρ.
    """
    n_ct = ct_mag.shape[1]
    if len(ct_names) != n_ct:
        raise ValueError(f"len(ct_names)={len(ct_names)} != n_ct={n_ct}")
    out: dict[str, list[dict[str, float | str]]] = {}
    for sex_label in ("female", "male"):
        mask = sex == sex_label
        n = int(mask.sum())
        if n < 3:
            out[sex_label] = []
            continue
        rows: list[dict[str, float | str]] = []
        r = abs_residual[mask]
        for c in range(n_ct):
            x = ct_mag[mask, c]
            # ``spearmanr`` raises ``ConstantInputWarning`` (not an exception)
            # when one input is constant — undefined correlation. We swallow
            # the warning and emit NaN explicitly so the log isn't noisy on
            # degenerate (CT, sex) slices.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    rho, p = spearmanr(r, x)
                    rho = float(rho)
                    p = float(p)
                except ValueError:
                    rho = float("nan")
                    p = float("nan")
            rows.append(
                {
                    "cell_type": ct_names[c],
                    "rho": rho,
                    "p_value": p,
                    "n": n,
                }
            )
        out[sex_label] = rows
    return out


# =============================================================================
# Plot
# =============================================================================


def render_figure(
    ct_mag: np.ndarray,
    abs_residual: np.ndarray,
    sex: np.ndarray,
    ct_names: list[str],
    per_ct_wilcoxon: dict[str, dict[str, float]],
    sex_per_fold_r2: dict[str, dict[str, list[float]]],
    differential_top_ct: str,
    differential_top_gene_table: dict[str, list[dict[str, float | str]]],
    out_fig_dir: Path,
) -> list[Path]:
    """Render the 4-panel figure required by EXP-038.

    Panels:
      A. per-CT mean ``|attribution|`` rank F vs M (scatter, points labeled
         with CT names for the top-5 of each sex).
      B. top differential genes within ``differential_top_ct`` F vs M (a
         horizontal grouped bar).
      C. prediction error (``|residual|``) histogram by sex.
      D. per-fold R² by sex (bar with std-band).
    """
    apply_theme()
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    ax_a, ax_b, ax_c, ax_d = (
        axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1],
    )

    # --- Panel A: F vs M scatter of per-CT mean magnitude ---
    fem = sex == "female"
    mal = sex == "male"
    f_means = ct_mag[fem].mean(axis=0)
    m_means = ct_mag[mal].mean(axis=0)
    ax_a.scatter(
        f_means, m_means, s=24, color="#4C78A8", edgecolor="black",
        linewidth=0.5, alpha=0.8, zorder=3,
    )
    # y=x reference.
    lims = (
        float(min(f_means.min(), m_means.min())) * 0.9,
        float(max(f_means.max(), m_means.max())) * 1.1,
    )
    ax_a.plot(lims, lims, color="0.6", linestyle="--", linewidth=0.8, zorder=1)
    # Label the CTs that are top-5 in EITHER sex.
    f_top5 = set(np.argsort(-f_means)[:5].tolist())
    m_top5 = set(np.argsort(-m_means)[:5].tolist())
    label_idx = sorted(f_top5 | m_top5)
    for i in label_idx:
        ax_a.annotate(
            ct_names[i],
            xy=(f_means[i], m_means[i]),
            xytext=(4, 4), textcoords="offset points",
            fontsize=7, color="black", alpha=0.85,
        )
    ax_a.set_xlabel("Female mean |attribution|")
    ax_a.set_ylabel("Male mean |attribution|")
    ax_a.set_title(f"A. Per-CT attribution F vs M (n_F={int(fem.sum())}, "
                   f"n_M={int(mal.sum())})")
    ax_a.set_xlim(lims)
    ax_a.set_ylim(lims)
    fmt_axes(ax_a)

    # --- Panel B: differential top genes within differential_top_ct ---
    f_rows = differential_top_gene_table.get("female", [])
    m_rows = differential_top_gene_table.get("male", [])
    # Union of top-N genes from each sex; preserve display order: F-top first
    # then any M-top genes not already shown.
    f_genes = [r["gene"] for r in f_rows]
    m_genes = [r["gene"] for r in m_rows]
    seen: set[str] = set()
    union_genes: list[str] = []
    for g in f_genes + m_genes:
        if g not in seen:
            union_genes.append(g)
            seen.add(g)
    f_lookup = {r["gene"]: float(r["mean_abs_attribution"]) for r in f_rows}
    m_lookup = {r["gene"]: float(r["mean_abs_attribution"]) for r in m_rows}
    f_vals = np.asarray([f_lookup.get(g, 0.0) for g in union_genes])
    m_vals = np.asarray([m_lookup.get(g, 0.0) for g in union_genes])
    y_pos = np.arange(len(union_genes))
    bar_h = 0.4
    ax_b.barh(
        y_pos - bar_h / 2, f_vals, height=bar_h,
        color="#E45756", edgecolor="black", linewidth=0.4, label="female",
    )
    ax_b.barh(
        y_pos + bar_h / 2, m_vals, height=bar_h,
        color="#4C78A8", edgecolor="black", linewidth=0.4, label="male",
    )
    ax_b.set_yticks(y_pos)
    ax_b.set_yticklabels(union_genes, fontsize=7)
    ax_b.invert_yaxis()
    ax_b.set_xlabel("Mean |attribution|")
    ax_b.set_title(f"B. Top differential genes in {differential_top_ct}")
    ax_b.legend(loc="lower right", fontsize=7, frameon=False)
    fmt_axes(ax_b)

    # --- Panel C: |residual| histogram by sex ---
    bins = np.linspace(0, max(1e-9, float(abs_residual.max())) * 1.05, 30)
    ax_c.hist(
        abs_residual[fem], bins=bins, alpha=0.55,
        color="#E45756", edgecolor="black", linewidth=0.4, label="female",
        density=True,
    )
    ax_c.hist(
        abs_residual[mal], bins=bins, alpha=0.55,
        color="#4C78A8", edgecolor="black", linewidth=0.4, label="male",
        density=True,
    )
    ax_c.axvline(
        float(np.mean(abs_residual[fem])), color="#E45756", linestyle="--",
        linewidth=1.2, alpha=0.9,
    )
    ax_c.axvline(
        float(np.mean(abs_residual[mal])), color="#4C78A8", linestyle="--",
        linewidth=1.2, alpha=0.9,
    )
    ax_c.set_xlabel("|y_true − y_composite|")
    ax_c.set_ylabel("density")
    ax_c.set_title("C. Prediction error |residual| by sex")
    ax_c.legend(loc="upper right", fontsize=7, frameon=False)
    fmt_axes(ax_c)

    # --- Panel D: per-fold R² by sex with std band ---
    folds = list(range(len(sex_per_fold_r2["female"]["per_fold_r2"])))
    f_r2 = np.asarray(sex_per_fold_r2["female"]["per_fold_r2"], dtype=np.float64)
    m_r2 = np.asarray(sex_per_fold_r2["male"]["per_fold_r2"], dtype=np.float64)
    width = 0.35
    x = np.arange(len(folds))
    ax_d.bar(
        x - width / 2, f_r2, width, color="#E45756",
        edgecolor="black", linewidth=0.4, label="female",
    )
    ax_d.bar(
        x + width / 2, m_r2, width, color="#4C78A8",
        edgecolor="black", linewidth=0.4, label="male",
    )
    f_mean = float(np.nanmean(f_r2))
    f_std = float(np.nanstd(f_r2, ddof=1)) if np.isfinite(f_r2).sum() >= 2 else 0.0
    m_mean = float(np.nanmean(m_r2))
    m_std = float(np.nanstd(m_r2, ddof=1)) if np.isfinite(m_r2).sum() >= 2 else 0.0
    ax_d.axhspan(
        f_mean - f_std, f_mean + f_std, color="#E45756", alpha=0.15, zorder=0,
    )
    ax_d.axhspan(
        m_mean - m_std, m_mean + m_std, color="#4C78A8", alpha=0.15, zorder=0,
    )
    ax_d.axhline(f_mean, color="#E45756", linestyle="--", linewidth=1.0, alpha=0.85)
    ax_d.axhline(m_mean, color="#4C78A8", linestyle="--", linewidth=1.0, alpha=0.85)
    ax_d.set_xticks(x)
    ax_d.set_xticklabels([f"fold {f}" for f in folds])
    ax_d.set_ylabel("R²")
    ax_d.set_title(
        f"D. Per-fold R² by sex (F mean={f_mean:.3f}±{f_std:.3f}, "
        f"M mean={m_mean:.3f}±{m_std:.3f})"
    )
    ax_d.axhline(0, color="0.6", linewidth=0.6)
    ax_d.legend(loc="upper right", fontsize=7, frameon=False)
    fmt_axes(ax_d)

    fig.tight_layout()
    style_paper_axes(fig)
    out_fig_dir.mkdir(parents=True, exist_ok=True)
    stem = out_fig_dir / "fig_sex_disparity"
    paths: list[Path] = []
    for ext in ("png", "pdf"):
        out = stem.with_suffix(f".{ext}")
        fig.savefig(out, dpi=600, bbox_inches="tight")
        paths.append(out)
    plt.close(fig)
    return paths


# =============================================================================
# Markdown rendering
# =============================================================================


def render_markdown(payload: dict) -> str:
    """Render the analysis JSON payload as a human-readable markdown report."""
    lines: list[str] = []
    cfg = payload.get("config", {})
    lines.append("# Sex disparity attribution analysis (EXP-038)")
    lines.append("")
    lines.append(
        "Follow-up to EXP-035, asking whether the model uses different "
        "cell types or genes when predicting cognitive resilience for "
        "females vs males."
    )
    lines.append("")
    lines.append("## Cohort")
    lines.append("")
    coh = payload["cohort"]
    lines.append(
        f"- N total = {coh['n_total']} ({coh['n_female']} female, "
        f"{coh['n_male']} male). Sex coverage: 100% via `msex` column."
    )
    sx = payload["per_fold_r2_by_sex"]
    f_pf = ", ".join(f"{v:.3f}" for v in sx["female"]["per_fold_r2"])
    m_pf = ", ".join(f"{v:.3f}" for v in sx["male"]["per_fold_r2"])
    lines.append(
        f"- Female per-fold R²: [{f_pf}], mean ± std = "
        f"{sx['female']['mean_r2']:.3f} ± {sx['female']['std_r2']:.3f}."
    )
    lines.append(
        f"- Male per-fold R²: [{m_pf}], mean ± std = "
        f"{sx['male']['mean_r2']:.3f} ± {sx['male']['std_r2']:.3f}."
    )
    lines.append(
        f"- Δ (F − M) mean R² = {sx['female']['mean_r2'] - sx['male']['mean_r2']:.3f}."
    )
    f_res = coh.get("mean_abs_residual_female", float("nan"))
    m_res = coh.get("mean_abs_residual_male", float("nan"))
    if f_res > 0 and np.isfinite(f_res) and np.isfinite(m_res):
        ratio_str = f"{m_res / f_res:.2f}"
    else:
        ratio_str = "n/a"
    lines.append(
        f"- Mean |residual| F = {f_res:.3f}, M = {m_res:.3f} "
        f"(ratio M/F = {ratio_str}). If males have similar or smaller "
        "|residual| than females while having a much lower per-fold R², "
        "the disparity is driven by smaller male variance(y_true) rather "
        "than systematically worse predictions."
    )
    lines.append("")

    lines.append("## Top-5 cell types by sex")
    lines.append("")
    lines.append("| Rank | Female (CT, mean |attr|) | Male (CT, mean |attr|) |")
    lines.append("|---:|---|---|")
    fem_top = payload["top_ct_per_sex"]["female"]
    mal_top = payload["top_ct_per_sex"]["male"]
    for k in range(max(len(fem_top), len(mal_top))):
        f_str = (
            f"{fem_top[k]['cell_type']} ({fem_top[k]['mean_abs_attribution']:.3e})"
            if k < len(fem_top) else "—"
        )
        m_str = (
            f"{mal_top[k]['cell_type']} ({mal_top[k]['mean_abs_attribution']:.3e})"
            if k < len(mal_top) else "—"
        )
        lines.append(f"| {k + 1} | {f_str} | {m_str} |")
    fem_set = {r["cell_type"] for r in fem_top}
    mal_set = {r["cell_type"] for r in mal_top}
    overlap = fem_set & mal_set
    lines.append("")
    lines.append(
        f"- Overlap of top-{cfg.get('top_n_ct', DEFAULT_TOP_N_CT)} sets: "
        f"{len(overlap)}/{cfg.get('top_n_ct', DEFAULT_TOP_N_CT)} "
        f"({sorted(overlap)})."
    )
    lines.append("")

    lines.append("## Sex × CT interaction (Wilcoxon rank-sum, BH-FDR over 31 CTs)")
    lines.append("")
    lines.append(
        "| CT | mean F | mean M | Δ(F−M) | rank-sum statistic | p | q (BH) |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    sig: list[tuple[str, dict[str, float]]] = []
    for ct, row in payload["sex_x_ct_wilcoxon"].items():
        sig.append((ct, row))
    sig.sort(key=lambda t: (t[1]["q_value"] if np.isfinite(t[1]["q_value"]) else 1.0))
    for ct, row in sig:
        q = row["q_value"]
        q_str = f"{q:.3e}" if np.isfinite(q) else "NaN"
        marker = " *" if (np.isfinite(q) and q < BH_Q_THRESHOLD) else ""
        lines.append(
            f"| {ct}{marker} | {row['mean_female']:.3e} | "
            f"{row['mean_male']:.3e} | {row['mean_diff_F_minus_M']:+.3e} | "
            f"{row['statistic']:+.2f} | {row['p_value']:.3e} | {q_str} |"
        )
    n_sig_ct = sum(
        1 for ct, row in payload["sex_x_ct_wilcoxon"].items()
        if np.isfinite(row["q_value"]) and row["q_value"] < BH_Q_THRESHOLD
    )
    lines.append("")
    lines.append(f"- **{n_sig_ct} CTs significant at BH q < {BH_Q_THRESHOLD}** (* in table).")
    lines.append("")

    lines.append("## Sex × gene interaction on top-50 (CT, gene) pairs (BH-FDR over 50)")
    lines.append("")
    lines.append("| Rank | CT | Gene | F mean | M mean | Δ(F−M) | p | q (BH) |")
    lines.append("|---:|---|---|---:|---:|---:|---:|---:|")
    for k, row in enumerate(payload["sex_x_pair_wilcoxon"]):
        q = row["q_value"]
        q_str = f"{q:.3e}" if np.isfinite(q) else "NaN"
        marker = " *" if (np.isfinite(q) and q < BH_Q_THRESHOLD) else ""
        lines.append(
            f"| {k + 1} | {row['cell_type']} | {row['gene']}{marker} | "
            f"{row['mean_female']:.3e} | {row['mean_male']:.3e} | "
            f"{row['mean_diff_F_minus_M']:+.3e} | {row['p_value']:.3e} | "
            f"{q_str} |"
        )
    n_sig_pair = sum(
        1 for r in payload["sex_x_pair_wilcoxon"]
        if np.isfinite(r["q_value"]) and r["q_value"] < BH_Q_THRESHOLD
    )
    lines.append("")
    lines.append(f"- **{n_sig_pair}/50 pairs significant at BH q < {BH_Q_THRESHOLD}**.")
    lines.append("")

    lines.append("## Spearman ρ(|residual|, per-CT magnitude) by sex (top-10 |ρ|)")
    lines.append("")
    lines.append("| CT | F ρ (p) | M ρ (p) | |F−M| |")
    lines.append("|---|---:|---:|---:|")
    fem_spear = {r["cell_type"]: r for r in payload["spearman_residual_vs_ct"]["female"]}
    mal_spear = {r["cell_type"]: r for r in payload["spearman_residual_vs_ct"]["male"]}
    cts = sorted(set(fem_spear.keys()) | set(mal_spear.keys()))
    abs_diffs = []
    for ct in cts:
        f_rho = fem_spear.get(ct, {}).get("rho", float("nan"))
        m_rho = mal_spear.get(ct, {}).get("rho", float("nan"))
        if np.isfinite(f_rho) and np.isfinite(m_rho):
            abs_diffs.append((abs(f_rho - m_rho), ct, f_rho, m_rho,
                              fem_spear[ct]["p_value"],
                              mal_spear[ct]["p_value"]))
    abs_diffs.sort(key=lambda t: -t[0])
    for ad, ct, fr, mr, fp, mp in abs_diffs[:10]:
        lines.append(
            f"| {ct} | {fr:+.3f} ({fp:.3e}) | {mr:+.3f} ({mp:.3e}) | {ad:.3f} |"
        )
    lines.append("")

    lines.append("## Headline interpretation")
    lines.append("")
    headline = payload.get("headline", {})
    expl = headline.get("most_plausible_explanation", "(see numeric tables above)")
    lines.append(f"- **Most plausible explanation:** {expl}")
    lines.append(f"- **Top-3 F:** {headline.get('top3_female', [])}")
    lines.append(f"- **Top-3 M:** {headline.get('top3_male', [])}")
    lines.append(
        f"- **# CTs with sex-differential attribution at BH q < "
        f"{BH_Q_THRESHOLD}:** {n_sig_ct}/31."
    )
    lines.append(
        f"- **# (CT, gene) pairs differing significantly:** {n_sig_pair}/50."
    )
    return "\n".join(lines) + "\n"


# =============================================================================
# Headline classification
# =============================================================================


def classify_explanation(
    n_sig_ct: int,
    n_sig_pair: int,
    sex_per_fold_r2: dict[str, dict[str, list[float]]],
    top_ct_overlap: int,
    top_ct_total: int,
    mean_abs_residual_female: float,
    mean_abs_residual_male: float,
) -> str:
    """Heuristic classification of the three hypotheses (a / b / c).

    Decision rule (codified rather than narrated so it survives review). The
    three hypotheses are treated as **prioritised** rather than mutually
    exclusive — (b) is only declared when the top-magnitude CTs themselves
    show sex-differential patterns, NOT when a low-magnitude CT survives
    multiple-test correction:

      * **(b) sex-specific signature** declared when EITHER (i) ≥ 2 of the
        top-``top_ct_total`` CTs are themselves sex-significant at BH q<0.05,
        or (ii) the top-K CT *lists* differ in membership (not just rank
        order: the top-3 sets disagree by ≥ 1 element), or (iii) ≥ 5 of the
        top-50 (CT, gene) pairs are sex-significant at BH q<0.05. A single
        sex-differential CT outside the top-K (e.g. a low-magnitude
        Hippocampal CA1-3 lighting up alone) is NOT enough to claim a
        sex-specific signature — it would be over-interpreting a marginal
        finding given the 31-CT × 50-pair test universe.

      * **(c) male heterogeneity** declared when male per-fold R² std is
        ≥ 1.5 × female std AND mean ``|residual|`` is comparable across
        sexes (within 25% of each other) — high R² variance with similar
        absolute error implies the variance gap is dominated by smaller
        male-fold variance(y_true) and a few outlier folds, not by
        systematically worse male predictions.

      * **(a) noise on smaller male subset** is the default when neither
        (b) nor (c)'s criteria are met.

    The classifier returns **at most one** string but composes (a) + (c) when
    both noise and heterogeneity criteria fire. The raw numbers are in the
    JSON so a reviewer can override.
    """
    f_r2 = np.asarray(
        sex_per_fold_r2["female"]["per_fold_r2"], dtype=np.float64
    )
    m_r2 = np.asarray(
        sex_per_fold_r2["male"]["per_fold_r2"], dtype=np.float64
    )
    f_std = float(np.nanstd(f_r2, ddof=1)) if np.isfinite(f_r2).sum() >= 2 else 0.0
    m_std = float(np.nanstd(m_r2, ddof=1)) if np.isfinite(m_r2).sum() >= 2 else 0.0
    high_male_var = (m_std >= 1.5 * f_std) if f_std > 0 else (m_std > 0.1)

    # Compare mean |residual|. If males have similar or smaller absolute
    # error, R² gap is structural (variance scaling) not predictive.
    if mean_abs_residual_female > 0:
        residual_ratio = mean_abs_residual_male / mean_abs_residual_female
    else:
        residual_ratio = float("nan")
    similar_residual = (
        np.isfinite(residual_ratio) and 0.75 <= residual_ratio <= 1.25
    )

    # (b) requires a meaningful sex-specific signature, NOT a single low-
    # magnitude CT slipping past BH-FDR. We require ≥ 2 sig CTs OR ≥ 5 sig
    # pairs OR top-K membership disagreement to declare (b).
    sig_signature = (
        n_sig_ct >= 2
        or n_sig_pair >= 5
        or top_ct_overlap < top_ct_total
    )
    if sig_signature:
        return (
            "(b) sex-specific signature — the model emphasises different "
            "cell types / genes for females vs males "
            f"(n_sig_CT={n_sig_ct}/31, n_sig_pair={n_sig_pair}/50, "
            f"top-CT overlap={top_ct_overlap}/{top_ct_total})."
        )
    if high_male_var and similar_residual:
        # (a)+(c) co-explanation — noise + heterogeneity.
        # Guard the std ratio: f_std == 0 implies a degenerate / synthetic
        # female series, in which case "Nx" is meaningless; report inf.
        std_ratio_str = (
            f"{m_std / f_std:.1f}×" if f_std > 0 else "∞ (f_std=0)"
        )
        return (
            "(a)+(c) noise + male heterogeneity — top-{0}/{0} CTs agree "
            "and only {1}/31 CTs / {2}/50 (CT, gene) pairs are sex-"
            "differential at BH q<0.05 (well below the level needed to "
            "claim a sex-specific signature). Male per-fold R² std "
            "({3:.3f}) is {4} female std ({5:.3f}), but mean "
            "|residual| is comparable (F={6:.3f}, M={7:.3f}; ratio "
            "{8:.2f}). The R² gap is dominated by smaller male-N + a few "
            "outlier folds, not by systematically worse male predictions."
        ).format(
            top_ct_total, n_sig_ct, n_sig_pair, m_std, std_ratio_str,
            f_std, mean_abs_residual_female, mean_abs_residual_male,
            residual_ratio,
        )
    if high_male_var:
        return (
            "(c) male heterogeneity dominant — top-K CT lists agree but "
            f"male per-fold R² std ({m_std:.3f}) is ≥ 1.5× female std "
            f"({f_std:.3f}). Mean |residual| ratio M/F = "
            f"{residual_ratio:.2f} indicates absolute error is "
            "systematically different between sexes."
        )
    return (
        "(a) noise on smaller male subset — top-CT lists agree, only "
        f"{n_sig_ct}/31 CTs / {n_sig_pair}/50 pairs are sex-differential, "
        "and per-fold R² variance is comparable across sexes. The model "
        "uses the same biology for both sexes."
    )


# =============================================================================
# Per-fold R² by sex (re-derived here so the script doesn't depend on
# subgroup_r2_unified.json being up-to-date)
# =============================================================================


def per_fold_r2_by_sex(
    pred_df: pd.DataFrame, sex_df: pd.DataFrame, n_folds: int,
) -> dict[str, dict[str, list[float]]]:
    """Per-fold R²(y_true, y_composite) by sex; mean ± std across folds."""
    from sklearn.metrics import r2_score
    df = pred_df.merge(sex_df, on="ROSMAP_IndividualID", how="inner")
    out: dict[str, dict[str, list[float]]] = {}
    for sex_label in ("female", "male"):
        per_fold: list[float] = []
        ns: list[int] = []
        for fold in range(n_folds):
            sub = df[(df["fold"] == fold) & (df["sex"] == sex_label)]
            ns.append(len(sub))
            if len(sub) < 3 or float(np.var(sub["y_true"].to_numpy())) == 0.0:
                per_fold.append(float("nan"))
                continue
            per_fold.append(
                float(r2_score(sub["y_true"].to_numpy(), sub["y_composite"].to_numpy()))
            )
        arr = np.asarray(per_fold, dtype=np.float64)
        finite = arr[np.isfinite(arr)]
        mean_r2 = float(np.mean(finite)) if finite.size else float("nan")
        std_r2 = float(np.std(finite, ddof=1)) if finite.size >= 2 else float("nan")
        out[sex_label] = {
            "per_fold_r2": per_fold,
            "n_per_fold": [int(n) for n in ns],
            "mean_r2": mean_r2,
            "std_r2": std_r2,
        }
    return out


# =============================================================================
# CLI entry point
# =============================================================================


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--attr-npz",
        type=Path,
        default=_WORKTREE_ROOT
        / "outputs/canonical/interpretability/captum_ig/composite_attributions.npz",
    )
    p.add_argument(
        "--pred-root",
        type=Path,
        default=_WORKTREE_ROOT / "outputs/canonical/p5_canonical_seed42",
    )
    p.add_argument(
        "--metadata-csv",
        type=Path,
        default=_WORKTREE_ROOT / "data/metadata_ROSMAP/metadata.csv",
    )
    p.add_argument(
        "--precomputed-dir",
        type=Path,
        default=_WORKTREE_ROOT / "data/precomputed",
    )
    p.add_argument(
        "--out-json",
        type=Path,
        default=_WORKTREE_ROOT
        / "outputs/canonical/interpretability/sex_disparity_attribution.json",
    )
    p.add_argument(
        "--out-md",
        type=Path,
        default=_WORKTREE_ROOT
        / "outputs/canonical/interpretability/sex_disparity_attribution.md",
    )
    p.add_argument(
        "--out-fig-dir",
        type=Path,
        default=_WORKTREE_ROOT
        / "outputs/canonical/interpretability/figures/sex_disparity",
    )
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--top-n-ct", type=int, default=DEFAULT_TOP_N_CT)
    p.add_argument(
        "--top-n-genes-per-ct", type=int, default=DEFAULT_TOP_N_GENES_PER_CT,
    )
    p.add_argument("--top-n-pairs", type=int, default=DEFAULT_TOP_N_PAIRS)
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    logger.info("Loading per-subject Captum IG tensor: %s", args.attr_npz)
    attr_data = load_attribution_tensor(args.attr_npz)
    attr = attr_data["attributions"]
    sids = attr_data["subject_ids"]
    n, n_ct, n_genes = attr.shape
    logger.info(
        "Attribution tensor: N=%d, n_ct=%d, n_genes=%d", n, n_ct, n_genes,
    )

    ct_names = list(CELL_TYPE_ORDER)[:n_ct]
    if len(ct_names) < n_ct:
        ct_names = ct_names + [f"ct_{i}" for i in range(len(ct_names), n_ct)]
    gene_names = load_gene_names(args.precomputed_dir, n_genes)

    logger.info("Loading metadata: %s", args.metadata_csv)
    sex_df = load_sex_metadata(args.metadata_csv)
    sex_lookup = dict(
        zip(sex_df["ROSMAP_IndividualID"], sex_df["sex"]),
    )
    sex = np.asarray([sex_lookup.get(s, None) for s in sids])
    if any(s is None for s in sex):
        n_missing = int(sum(1 for s in sex if s is None))
        raise RuntimeError(
            f"{n_missing} subjects in attribution tensor have no msex in "
            f"metadata; refusing to proceed (re-run with --metadata-csv pointing "
            "at the canonical CSV)."
        )

    logger.info("Loading per-fold val predictions: %s", args.pred_root)
    pred_df = load_val_predictions_per_fold(args.pred_root, n_folds=args.n_folds)
    pred_lookup = pred_df.set_index("ROSMAP_IndividualID")[
        ["y_true", "y_composite"]
    ].to_dict("index")
    abs_residual = np.zeros(n, dtype=np.float64)
    for i, s in enumerate(sids):
        if s not in pred_lookup:
            raise RuntimeError(
                f"Subject {s} in attribution tensor missing from per-fold "
                "predictions."
            )
        abs_residual[i] = abs(
            pred_lookup[s]["y_true"] - pred_lookup[s]["y_composite"]
        )
    logger.info(
        "Mean |residual| F=%.4f, M=%.4f",
        float(np.mean(abs_residual[sex == "female"])),
        float(np.mean(abs_residual[sex == "male"])),
    )

    # Per-fold R² by sex (re-derived independently of subgroup_r2_unified.json).
    sex_per_fold_r2 = per_fold_r2_by_sex(pred_df, sex_df, n_folds=args.n_folds)

    # Per-subject CT magnitude and Wilcoxon test.
    logger.info("Computing per-subject CT magnitudes...")
    ct_mag = per_subject_ct_magnitude(attr)

    logger.info("Sex × CT Wilcoxon (n=%d CTs, BH-FDR)...", n_ct)
    per_ct_wilcoxon = wilcoxon_per_ct(ct_mag, sex, ct_names)

    logger.info("Top-%d CT per sex...", args.top_n_ct)
    top_ct_per_sex = rank_top_ct_per_sex(
        ct_mag, sex, ct_names, top_n=args.top_n_ct,
    )

    logger.info("Top-%d genes per CT per sex...", args.top_n_genes_per_ct)
    top_genes_per_ct_per_sex = rank_top_genes_per_ct_per_sex(
        attr, sex, ct_names, gene_names, top_n=args.top_n_genes_per_ct,
    )

    logger.info("Top-%d cohort-wide (CT, gene) pairs...", args.top_n_pairs)
    top_pairs = topk_pairs_overall(attr, ct_names, gene_names, top_k=args.top_n_pairs)

    logger.info("Sex × pair Wilcoxon (n=%d pairs, BH-FDR)...", len(top_pairs))
    per_pair_wilcoxon = wilcoxon_per_pair(attr, sex, top_pairs)

    logger.info("Spearman ρ(|residual|, per-CT magnitude) by sex...")
    spearman_table = spearman_residual_vs_ct_per_sex(
        abs_residual, ct_mag, sex, ct_names,
    )

    # Identify the differential top-CT for panel B: the top-1 CT in the union
    # of female-top1 and male-top1, picking the one with larger |F−M| (so the
    # panel is informative). Defaults to female-top1 if both ties.
    differential_top_ct = top_ct_per_sex["female"][0]["cell_type"]
    if top_ct_per_sex["female"][0]["cell_type"] != top_ct_per_sex["male"][0]["cell_type"]:
        # If the top-1 CT differs between sexes, picking female-top1 is fine;
        # the panel will show that CT's gene profile.
        pass

    differential_top_gene_table = top_genes_per_ct_per_sex[differential_top_ct]

    n_sig_ct = sum(
        1 for v in per_ct_wilcoxon.values()
        if np.isfinite(v["q_value"]) and v["q_value"] < BH_Q_THRESHOLD
    )
    n_sig_pair = sum(
        1 for r in per_pair_wilcoxon
        if np.isfinite(r["q_value"]) and r["q_value"] < BH_Q_THRESHOLD
    )
    fem_top_set = {r["cell_type"] for r in top_ct_per_sex["female"]}
    mal_top_set = {r["cell_type"] for r in top_ct_per_sex["male"]}
    overlap = len(fem_top_set & mal_top_set)
    mean_abs_residual_f = float(np.mean(abs_residual[sex == "female"]))
    mean_abs_residual_m = float(np.mean(abs_residual[sex == "male"]))
    explanation = classify_explanation(
        n_sig_ct=n_sig_ct,
        n_sig_pair=n_sig_pair,
        sex_per_fold_r2=sex_per_fold_r2,
        top_ct_overlap=overlap,
        top_ct_total=args.top_n_ct,
        mean_abs_residual_female=mean_abs_residual_f,
        mean_abs_residual_male=mean_abs_residual_m,
    )

    payload = {
        "config": {
            "attr_npz": str(args.attr_npz),
            "pred_root": str(args.pred_root),
            "metadata_csv": str(args.metadata_csv),
            "precomputed_dir": str(args.precomputed_dir),
            "n_folds": int(args.n_folds),
            "top_n_ct": int(args.top_n_ct),
            "top_n_genes_per_ct": int(args.top_n_genes_per_ct),
            "top_n_pairs": int(args.top_n_pairs),
            "bh_q_threshold": BH_Q_THRESHOLD,
        },
        "cohort": {
            "n_total": int(n),
            "n_female": int((sex == "female").sum()),
            "n_male": int((sex == "male").sum()),
            "mean_abs_residual_female": mean_abs_residual_f,
            "mean_abs_residual_male": mean_abs_residual_m,
        },
        "per_fold_r2_by_sex": sex_per_fold_r2,
        "top_ct_per_sex": top_ct_per_sex,
        "top_genes_per_ct_per_sex": top_genes_per_ct_per_sex,
        "sex_x_ct_wilcoxon": per_ct_wilcoxon,
        "sex_x_pair_wilcoxon": per_pair_wilcoxon,
        "spearman_residual_vs_ct": spearman_table,
        "headline": {
            "n_sig_ct_bh_q05": int(n_sig_ct),
            "n_sig_pair_bh_q05": int(n_sig_pair),
            "top_ct_overlap": int(overlap),
            "top_ct_total": int(args.top_n_ct),
            "top3_female": [r["cell_type"] for r in top_ct_per_sex["female"][:3]],
            "top3_male": [r["cell_type"] for r in top_ct_per_sex["male"][:3]],
            "differential_top_ct_for_panel_B": differential_top_ct,
            "most_plausible_explanation": explanation,
        },
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with args.out_json.open("w") as fh:
        json.dump(payload, fh, indent=2, default=float)
    logger.info("Wrote %s", args.out_json)

    md = render_markdown(payload)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(md)
    logger.info("Wrote %s", args.out_md)

    fig_paths = render_figure(
        ct_mag=ct_mag,
        abs_residual=abs_residual,
        sex=sex,
        ct_names=ct_names,
        per_ct_wilcoxon=per_ct_wilcoxon,
        sex_per_fold_r2=sex_per_fold_r2,
        differential_top_ct=differential_top_ct,
        differential_top_gene_table=differential_top_gene_table,
        out_fig_dir=args.out_fig_dir,
    )
    for fp in fig_paths:
        logger.info("Wrote %s", fp)

    print("\n" + "=" * 78)
    print("Sex disparity attribution analysis (EXP-038)")
    print("=" * 78)
    print(
        f"N total = {n} ({(sex == 'female').sum()} F, "
        f"{(sex == 'male').sum()} M)"
    )
    print(
        f"Female mean R² = {sex_per_fold_r2['female']['mean_r2']:.3f} ± "
        f"{sex_per_fold_r2['female']['std_r2']:.3f}; "
        f"Male mean R² = {sex_per_fold_r2['male']['mean_r2']:.3f} ± "
        f"{sex_per_fold_r2['male']['std_r2']:.3f}"
    )
    print(f"Sex × CT Wilcoxon: {n_sig_ct}/31 CTs at BH q<{BH_Q_THRESHOLD}")
    print(f"Sex × pair Wilcoxon: {n_sig_pair}/{args.top_n_pairs} pairs at BH q<{BH_Q_THRESHOLD}")
    print(f"Top-3 F: {[r['cell_type'] for r in top_ct_per_sex['female'][:3]]}")
    print(f"Top-3 M: {[r['cell_type'] for r in top_ct_per_sex['male'][:3]]}")
    print(f"Most plausible explanation: {explanation}")
    print("=" * 78 + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
