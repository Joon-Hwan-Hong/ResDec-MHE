#!/usr/bin/env python
"""Unified subgroup R² table for ResDec-MHE composite predictions.

Per-fold (NOT pooled across folds) subgroup metrics for five demographic
strata:

1. **APOE-ε4 dosage** — count of ε4 alleles in ``apoe_genotype`` (0 / 1 / 2).
2. **Sex** — ``msex`` (0 = female, 1 = male).
3. **Age tertiles** — cohort-wide tertile cuts on ``age_death`` (T1 = lowest,
   T2 = middle, T3 = highest).
4. **Education tertiles** — cohort-wide tertile cuts on ``educ`` (T1 = lowest,
   T2 = middle, T3 = highest).
5. **AD-dx** — ``cogdx ∈ {4, 5}`` vs ``cogdx ∈ {1, 2, 3}`` (excludes
   ``cogdx == 6`` "Other dementia" from this binary contrast).

For each (subgroup, stratum) we compute, **per fold separately**:

  * ``n``        — number of subjects in that stratum within the fold
  * ``r2``       — R² of the composite prediction (``y_hat = y_tabpfn + f₁``)
                   against ``y_true``
  * ``pearson_r`` — Pearson correlation
  * ``mae``      — mean absolute error

Aggregations across the 5 folds:

  * ``n_total``      — sum of per-fold ``n`` (each subject appears in exactly
                       one val fold under k-fold CV)
  * ``mean_r2``      — arithmetic mean of per-fold R² over folds with valid n
  * ``std_r2``       — sample std (ddof=1) when ≥ 2 folds, else NaN
  * Same for ``mean_pearson_r`` / ``std_pearson_r`` / ``mean_mae`` /
    ``std_mae``.

Tertile cuts are computed **cohort-wide** (over the union of all 5 val folds)
on the non-null subset of the covariate, then applied uniformly to every fold
so the same age/education boundary is used in fold 0 and fold 4.

Inputs (defaults; CLI-overridable):
    --pred-root      outputs/canonical/p5_canonical_seed42
    --tabpfn-dir     data/canonical
    --metadata-csv   data/metadata_ROSMAP/metadata.csv

Outputs:
    --out-json       outputs/canonical/interpretability/subgroup_r2_unified.json
    --out-md         outputs/canonical/interpretability/subgroup_r2_unified.md
    --out-fig-dir    outputs/canonical/interpretability/figures/subgroup_r2/
                       fig_subgroup_r2.{png,pdf}

Example::

    PYTHONPATH=<worktree-root> uv run python \\
        scripts/resdec_mhe/interpretability/run_subgroup_r2_table.py
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # must be before pyplot import
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import r2_score

_WORKTREE_ROOT = Path(__file__).resolve().parents[3]
if str(_WORKTREE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKTREE_ROOT))

from src.analysis.resdec_io import load_all_folds  # noqa: E402
from src.analysis.subgroup_helpers import apoe_e4_count_label  # noqa: E402
from src.visualization.theme import apply_theme, fmt_axes, style_paper_axes  # noqa: E402

logger = logging.getLogger(__name__)

# AD-dx encoding: ROSMAP cogdx 4=AD-probable, 5=AD-possible (combined as "AD"),
# 1/2/3 = NCI/MCI/MCI+ (combined as "non-AD"); 6=Other dementia is excluded
# from this binary contrast.
AD_COGDX_CODES: frozenset[float] = frozenset({4.0, 5.0})
NONAD_COGDX_CODES: frozenset[float] = frozenset({1.0, 2.0, 3.0})

# Per-fold metric is undefined for fewer than 3 paired points (Pearson
# correlation is ill-defined on n<3 and a 2-point R² is uninformative).
MIN_N_PER_FOLD_FOR_METRICS = 3

# Metadata columns we pull for the join. Listing them explicitly so a rename
# in the source CSV surfaces as an immediate KeyError instead of silently
# producing all-NaN strata.
_METADATA_COLS: tuple[str, ...] = (
    "ROSMAP_IndividualID",
    "apoe_genotype",
    "msex",
    "age_death",
    "educ",
    "cogdx",
)


def _tertile_label(value: float, cuts: tuple[float, float]) -> str | None:
    """Map a numeric ``value`` to ``T1`` / ``T2`` / ``T3`` using ``cuts``.

    Boundaries follow the ``np.quantile`` convention with ``[1/3, 2/3]``: a
    value strictly below ``cuts[0]`` is ``T1``, strictly above ``cuts[1]`` is
    ``T3``, and equal-or-between is ``T2``. Equality with a cut point is
    assigned to the upper bucket so the tertile ordering is well-defined.
    """
    if not np.isfinite(value):
        return None
    if value < cuts[0]:
        return "T1"
    if value < cuts[1]:
        return "T2"
    return "T3"


def _ad_dx_label(cogdx: float) -> str | None:
    """Binarize ``cogdx`` to ``"AD"`` / ``"non-AD"`` or None.

    ``cogdx == 6`` (Other dementia) and missing values are excluded from the
    binary contrast and return ``None``.
    """
    if not np.isfinite(cogdx):
        return None
    cog = float(cogdx)
    if cog in AD_COGDX_CODES:
        return "AD"
    if cog in NONAD_COGDX_CODES:
        return "non-AD"
    return None


def _compute_tertile_cuts(series: pd.Series) -> tuple[float, float]:
    """Return cohort-wide ``(q1/3, q2/3)`` tertile cuts on the non-null subset.

    Uses ``np.quantile`` (linear interpolation, default method) for exact
    equal-mass tertiles. Caller is responsible for ensuring the input series
    has at least 3 non-null values; a smaller series will still return cut
    values but the resulting tertile assignment will be degenerate.
    """
    valid = series.dropna().to_numpy(dtype=np.float64)
    if valid.size < 3:
        raise ValueError(
            f"Cannot compute tertiles on n={valid.size} non-null values "
            "(need ≥ 3)."
        )
    q1, q2 = np.quantile(valid, [1 / 3, 2 / 3])
    return float(q1), float(q2)


def assign_strata(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, dict[str, float | tuple[float, float]]]]:
    """Add ``stratum_*`` columns for the five subgroup families.

    Returns ``(df_with_strata, stratum_metadata)`` where ``stratum_metadata``
    records, per family, the tertile cuts (or NaN for non-quantile families)
    so the caller can write them to JSON for reproducibility.

    The five new columns added to ``df``:
      * ``stratum_apoe``  — ``"0"`` / ``"1"`` / ``"2"`` or NaN
      * ``stratum_sex``   — ``"female"`` / ``"male"`` or NaN
      * ``stratum_age``   — ``"T1"`` / ``"T2"`` / ``"T3"`` or NaN
      * ``stratum_educ``  — ``"T1"`` / ``"T2"`` / ``"T3"`` or NaN
      * ``stratum_addx``  — ``"AD"`` / ``"non-AD"`` or NaN
    """
    out = df.copy()

    # APOE-ε4 dosage from genotype string.
    out["stratum_apoe"] = out["apoe_genotype"].apply(apoe_e4_count_label)

    # Sex with explicit string labels.
    def _sex(x: object) -> str | None:
        if pd.isna(x):
            return None
        try:
            return "female" if int(x) == 0 else "male"
        except (TypeError, ValueError):
            return None

    out["stratum_sex"] = out["msex"].apply(_sex)

    # Age tertiles — cohort-wide cut.
    age_cuts = _compute_tertile_cuts(out["age_death"])
    out["stratum_age"] = out["age_death"].apply(
        lambda v: _tertile_label(float(v) if pd.notna(v) else float("nan"), age_cuts)
    )

    # Education tertiles — cohort-wide cut.
    educ_cuts = _compute_tertile_cuts(out["educ"])
    out["stratum_educ"] = out["educ"].apply(
        lambda v: _tertile_label(float(v) if pd.notna(v) else float("nan"), educ_cuts)
    )

    # AD-dx binary contrast.
    out["stratum_addx"] = out["cogdx"].apply(
        lambda v: _ad_dx_label(float(v) if pd.notna(v) else float("nan"))
    )

    metadata = {
        "apoe": {"description": "Count of ε4 alleles in apoe_genotype"},
        "sex": {"description": "msex (0=female, 1=male)"},
        "age": {
            "description": "age_death tertiles (cohort-wide cuts)",
            "tertile_cuts_q1_q2": age_cuts,
        },
        "educ": {
            "description": "educ tertiles (cohort-wide cuts)",
            "tertile_cuts_q1_q2": educ_cuts,
        },
        "addx": {
            "description": (
                "Binary AD-dx: cogdx ∈ {4, 5} → 'AD'; cogdx ∈ {1, 2, 3} → "
                "'non-AD'; cogdx == 6 (Other dementia) excluded."
            ),
        },
    }
    return out, metadata


def _per_fold_metrics(
    df: pd.DataFrame, n_folds: int, stratum_col: str, stratum_value: str,
) -> dict[str, list[float]]:
    """Per-fold ``r2`` / ``pearson_r`` / ``mae`` / ``n`` for one (col, value).

    Returns a dict with four lists each of length ``n_folds``. Folds whose
    stratum slice has fewer than ``MIN_N_PER_FOLD_FOR_METRICS`` subjects (or
    zero variance in ``y_true`` → undefined R²/Pearson) yield NaN for the
    three metrics; ``n`` is the integer count regardless. NaN values are
    later filtered when computing the mean ± std across folds.
    """
    r2s: list[float] = []
    rs: list[float] = []
    maes: list[float] = []
    ns: list[float] = []
    for fold in range(n_folds):
        sub = df[(df["fold"] == fold) & (df[stratum_col] == stratum_value)]
        n = len(sub)
        ns.append(float(n))
        if n < MIN_N_PER_FOLD_FOR_METRICS:
            r2s.append(float("nan"))
            rs.append(float("nan"))
            maes.append(float("nan"))
            continue
        y_true = sub["y_true"].to_numpy(dtype=np.float64)
        y_pred = sub["y_composite"].to_numpy(dtype=np.float64)
        # MAE is always defined for n ≥ 1.
        mae = float(np.mean(np.abs(y_pred - y_true)))
        maes.append(mae)
        # R²/Pearson require non-degenerate y_true variance.
        if np.var(y_true) == 0.0:
            r2s.append(float("nan"))
            rs.append(float("nan"))
            continue
        r2s.append(float(r2_score(y_true, y_pred)))
        # pearsonr returns nan for constant inputs; we already gated on var=0
        # for y_true. Constant y_pred → r=nan from scipy, propagate as-is.
        try:
            rs.append(float(pearsonr(y_true, y_pred).statistic))
        except ValueError:
            rs.append(float("nan"))
    return {"r2": r2s, "pearson_r": rs, "mae": maes, "n": ns}


def _aggregate(per_fold: list[float]) -> tuple[float, float]:
    """Mean and sample-std (ddof=1) over finite entries; std=NaN if <2 finite."""
    arr = np.asarray(per_fold, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("nan"), float("nan")
    mean = float(np.mean(finite))
    std = float(np.std(finite, ddof=1)) if finite.size >= 2 else float("nan")
    return mean, std


def compute_subgroup_r2_table(
    df: pd.DataFrame, n_folds: int,
) -> dict[str, dict[str, dict]]:
    """Build the nested per-fold subgroup metrics dict.

    Output structure (one entry per (family, stratum)):

        {
          "apoe": {
            "0": {
              "n_total": int, "n_per_fold": [n0, ..., n4],
              "per_fold_r2": [...], "mean_r2": float, "std_r2": float,
              "per_fold_pearson_r": [...], "mean_pearson_r": float, "std_pearson_r": float,
              "per_fold_mae": [...], "mean_mae": float, "std_mae": float,
            }, ...
          },
          ...
        }
    """
    families = {
        "apoe": ("stratum_apoe", ["0", "1", "2"]),
        "sex": ("stratum_sex", ["female", "male"]),
        "age": ("stratum_age", ["T1", "T2", "T3"]),
        "educ": ("stratum_educ", ["T1", "T2", "T3"]),
        "addx": ("stratum_addx", ["non-AD", "AD"]),
    }

    out: dict[str, dict[str, dict]] = {}
    for family, (col, strata) in families.items():
        out[family] = {}
        for stratum in strata:
            metrics = _per_fold_metrics(df, n_folds, col, stratum)
            mean_r2, std_r2 = _aggregate(metrics["r2"])
            mean_r, std_r = _aggregate(metrics["pearson_r"])
            mean_mae, std_mae = _aggregate(metrics["mae"])
            n_per_fold = [int(x) for x in metrics["n"]]
            n_total = int(sum(n_per_fold))
            out[family][stratum] = {
                "n_total": n_total,
                "n_per_fold": n_per_fold,
                "per_fold_r2": metrics["r2"],
                "mean_r2": mean_r2,
                "std_r2": std_r2,
                "per_fold_pearson_r": metrics["pearson_r"],
                "mean_pearson_r": mean_r,
                "std_pearson_r": std_r,
                "per_fold_mae": metrics["mae"],
                "mean_mae": mean_mae,
                "std_mae": std_mae,
            }
    return out


def _format_per_fold(values: list[float]) -> str:
    """Format a per-fold list as ``[v0, v1, …]`` with NaN spelled out."""
    parts = []
    for v in values:
        if isinstance(v, float) and not np.isfinite(v):
            parts.append("NaN")
        else:
            parts.append(f"{float(v):.3f}")
    return "[" + ", ".join(parts) + "]"


def render_markdown_table(
    table: dict[str, dict[str, dict]],
    canonical_r2_per_fold: list[float] | None = None,
) -> str:
    """Render ``table`` as a markdown report with one row per (family, stratum).

    Adds an "Overall" row at the top using ``canonical_r2_per_fold`` (the
    overall per-fold R² of the composite, not subgroup-restricted) when
    provided. Returns the markdown string ready to write to disk.
    """
    family_titles = {
        "apoe": "APOE-ε4 dosage",
        "sex": "Sex",
        "age": "Age at death (tertiles)",
        "educ": "Education (tertiles)",
        "addx": "AD diagnosis (cogdx)",
    }
    stratum_pretty = {
        "0": "ε4=0", "1": "ε4=1", "2": "ε4=2",
        "female": "female", "male": "male",
        "T1": "T1 (low)", "T2": "T2 (mid)", "T3": "T3 (high)",
        "non-AD": "non-AD (cogdx∈{1,2,3})",
        "AD": "AD (cogdx∈{4,5})",
    }
    lines: list[str] = []
    lines.append("# Subgroup R² unified table (per-fold)")
    lines.append("")
    lines.append(
        "Per-fold R² of composite prediction (`y_hat = y_tabpfn + f_1`) "
        "against `y_true`, sliced by demographic strata. Tertiles are "
        "cohort-wide cuts; AD-dx is `cogdx ∈ {4, 5}` vs `cogdx ∈ {1, 2, 3}` "
        "(`cogdx == 6` excluded)."
    )
    lines.append("")
    lines.append(
        "| Subgroup | Stratum | n | per-fold R² | mean R² ± std | mean Pearson r ± std | mean MAE ± std |"
    )
    lines.append("|---|---|---:|---|---|---|---|")

    if canonical_r2_per_fold is not None:
        mean_r2, std_r2 = _aggregate(canonical_r2_per_fold)
        n_total = "—"
        lines.append(
            f"| Overall | (all subjects) | {n_total} | "
            f"{_format_per_fold(canonical_r2_per_fold)} | "
            f"{mean_r2:.3f} ± {std_r2:.3f} | — | — |"
        )

    for family, strata in table.items():
        title = family_titles.get(family, family)
        for stratum, stats in strata.items():
            pretty = stratum_pretty.get(stratum, stratum)
            mean_r = stats["mean_pearson_r"]
            std_r = stats["std_pearson_r"]
            mean_m = stats["mean_mae"]
            std_m = stats["std_mae"]
            mean_r2 = stats["mean_r2"]
            std_r2 = stats["std_r2"]
            lines.append(
                f"| {title} | {pretty} | {stats['n_total']} | "
                f"{_format_per_fold(stats['per_fold_r2'])} | "
                f"{mean_r2:.3f} ± "
                f"{('NaN' if not np.isfinite(std_r2) else f'{std_r2:.3f}')} | "
                f"{mean_r:.3f} ± "
                f"{('NaN' if not np.isfinite(std_r) else f'{std_r:.3f}')} | "
                f"{mean_m:.3f} ± "
                f"{('NaN' if not np.isfinite(std_m) else f'{std_m:.3f}')} |"
            )

    # Headline summary lines.
    flat: list[tuple[str, str, dict]] = []
    for fam, strata in table.items():
        for stratum, stats in strata.items():
            flat.append((fam, stratum, stats))
    if flat:
        smallest = min(flat, key=lambda r: r[2]["n_total"])
        with_finite = [r for r in flat if np.isfinite(r[2]["mean_r2"])]
        if with_finite:
            highest = max(with_finite, key=lambda r: r[2]["mean_r2"])
            lowest = min(with_finite, key=lambda r: r[2]["mean_r2"])
            lines.append("")
            lines.append("## Notes")
            lines.append("")
            lines.append(
                f"- **Smallest stratum:** "
                f"`{smallest[0]}::{smallest[1]}` "
                f"(n={smallest[2]['n_total']}, mean R²="
                f"{smallest[2]['mean_r2']:.3f}). Treat single-fold metrics "
                f"with caution when fold-wise n < {MIN_N_PER_FOLD_FOR_METRICS}."
            )
            lines.append(
                f"- **Highest mean R²:** "
                f"`{highest[0]}::{highest[1]}` "
                f"(mean R²={highest[2]['mean_r2']:.3f}, "
                f"n={highest[2]['n_total']})."
            )
            lines.append(
                f"- **Lowest mean R²:** "
                f"`{lowest[0]}::{lowest[1]}` "
                f"(mean R²={lowest[2]['mean_r2']:.3f}, "
                f"n={lowest[2]['n_total']})."
            )
    return "\n".join(lines) + "\n"


def render_forest_plot(
    table: dict[str, dict[str, dict]],
    canonical_r2_per_fold: list[float] | None,
    out_fig_dir: Path,
) -> list[Path]:
    """Render a 5-panel grouped bar/forest figure of mean ± std R² per stratum.

    One panel per family (APOE / sex / age / educ / AD-dx). Each panel plots
    a horizontal bar per stratum at ``mean_r2`` with error caps at ± std (when
    ≥ 2 folds have finite metrics). Per-fold dots are overlaid in muted gray
    so the reader sees the underlying spread, and the overall canonical R²
    (mean over all 5 folds, not subgroup-restricted) is drawn as a vertical
    reference line on every panel.

    Saves both PNG and PDF at 600 DPI via ``save_fig`` and returns the list of
    written paths.
    """
    apply_theme()

    family_order = ["apoe", "sex", "age", "educ", "addx"]
    family_titles = {
        "apoe": "APOE-ε4 dosage",
        "sex": "Sex",
        "age": "Age tertiles",
        "educ": "Education tertiles",
        "addx": "AD diagnosis",
    }
    stratum_pretty = {
        "0": "ε4=0", "1": "ε4=1", "2": "ε4=2",
        "female": "female", "male": "male",
        "T1": "T1 (low)", "T2": "T2 (mid)", "T3": "T3 (high)",
        "non-AD": "non-AD", "AD": "AD",
    }

    fig, axes = plt.subplots(1, 5, figsize=(16, 4.5), sharex=True)
    if canonical_r2_per_fold is not None:
        canonical_mean, _ = _aggregate(canonical_r2_per_fold)
    else:
        canonical_mean = None

    for ax, fam in zip(axes, family_order):
        strata = table.get(fam, {})
        labels = list(strata.keys())
        means = np.array(
            [strata[s]["mean_r2"] for s in labels], dtype=np.float64
        )
        stds = np.array(
            [strata[s]["std_r2"] for s in labels], dtype=np.float64
        )
        ns = [strata[s]["n_total"] for s in labels]
        per_fold_lists = [strata[s]["per_fold_r2"] for s in labels]
        y_pos = np.arange(len(labels))[::-1]  # top-to-bottom in display order

        # NaN std → 0 error bar (single-fold case) so matplotlib still draws
        # the bar; the markdown table reports the underlying NaN.
        plot_stds = np.where(np.isfinite(stds), stds, 0.0)
        ax.barh(
            y_pos, means, xerr=plot_stds,
            color="#4C78A8", alpha=0.75, edgecolor="black",
            error_kw={"ecolor": "black", "capsize": 4, "lw": 1.0},
        )
        # Overlay per-fold dots in muted gray.
        for i, fold_vals in enumerate(per_fold_lists):
            arr = np.asarray(fold_vals, dtype=np.float64)
            finite = arr[np.isfinite(arr)]
            if finite.size:
                ax.scatter(
                    finite, np.full(finite.size, y_pos[i]),
                    color="0.3", s=12, zorder=3, alpha=0.8,
                    edgecolor="white", linewidth=0.4,
                )

        if canonical_mean is not None:
            ax.axvline(
                canonical_mean, color="crimson", linestyle="--",
                linewidth=1.2, alpha=0.8, label=f"overall R²={canonical_mean:.3f}",
            )

        ax.set_yticks(y_pos)
        ax.set_yticklabels(
            [f"{stratum_pretty.get(lbl, lbl)} (n={n})" for lbl, n in zip(labels, ns)]
        )
        ax.set_title(family_titles.get(fam, fam))
        ax.set_xlabel("R² (per-fold mean ± std)")
        ax.axvline(0.0, color="0.7", linewidth=0.6, zorder=0)
        fmt_axes(ax)

    # Single shared legend for the canonical line.
    if canonical_mean is not None:
        axes[-1].legend(
            loc="lower right", fontsize=8, frameon=False,
        )

    fig.tight_layout()
    out_fig_dir.mkdir(parents=True, exist_ok=True)
    # Apply the project's paper-style axes treatment (drop top/right ticks +
    # spines) at the same single chokepoint ``save_fig`` would, then write
    # both PNG and PDF directly: the project ``save_fig`` defaults to PNG-only
    # for lab-meeting deliverables, but this script's spec explicitly asks
    # for both PNG and PDF, so we bypass that helper and write each format
    # ourselves at the same 600 DPI.
    style_paper_axes(fig)
    stem = out_fig_dir / "fig_subgroup_r2"
    paths: list[Path] = []
    for ext in ("png", "pdf"):
        out = stem.with_suffix(f".{ext}")
        fig.savefig(out, dpi=600, bbox_inches="tight")
        paths.append(out)
    plt.close(fig)
    return paths


def _print_summary(
    table: dict[str, dict[str, dict]], canonical_per_fold: list[float] | None,
) -> None:
    """Human-readable stdout summary for the orchestrator log."""
    print("\n" + "=" * 78)
    print("Subgroup R² unified table — per-fold composite predictions")
    print("=" * 78)
    if canonical_per_fold is not None:
        m, s = _aggregate(canonical_per_fold)
        print(
            f"Overall (all subjects): per-fold R²={_format_per_fold(canonical_per_fold)} "
            f"  →  mean ± std = {m:.3f} ± "
            f"{'NaN' if not np.isfinite(s) else f'{s:.3f}'}"
        )
    print("-" * 78)
    print(
        f"  {'family/stratum':<25s}  {'n':>5s}  {'mean R²':>9s}  "
        f"{'std R²':>8s}  {'mean MAE':>9s}"
    )
    for fam, strata in table.items():
        for stratum, stats in strata.items():
            std_r2 = stats["std_r2"]
            std_str = f"{std_r2:8.3f}" if np.isfinite(std_r2) else "     NaN"
            print(
                f"  {f'{fam}/{stratum}':<25s}  {stats['n_total']:5d}  "
                f"{stats['mean_r2']:9.3f}  {std_str}  "
                f"{stats['mean_mae']:9.3f}"
            )
    print("=" * 78 + "\n")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--pred-root",
        type=Path,
        default=_WORKTREE_ROOT / "outputs/canonical/p5_canonical_seed42",
        help="Directory with fold{0..N-1}/val_predictions_best.npz",
    )
    p.add_argument(
        "--tabpfn-dir",
        type=Path,
        default=_WORKTREE_ROOT / "data/canonical",
        help="Directory with tabpfn_outer_fold{0..N-1}.npz",
    )
    p.add_argument(
        "--metadata-csv",
        type=Path,
        default=_WORKTREE_ROOT / "data/metadata_ROSMAP/metadata.csv",
        help="ROSMAP metadata CSV.",
    )
    p.add_argument(
        "--out-json",
        type=Path,
        default=_WORKTREE_ROOT
        / "outputs/canonical/interpretability/subgroup_r2_unified.json",
    )
    p.add_argument(
        "--out-md",
        type=Path,
        default=_WORKTREE_ROOT
        / "outputs/canonical/interpretability/subgroup_r2_unified.md",
    )
    p.add_argument(
        "--out-fig-dir",
        type=Path,
        default=_WORKTREE_ROOT
        / "outputs/canonical/interpretability/figures/subgroup_r2",
    )
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    for path in (args.pred_root, args.tabpfn_dir, args.metadata_csv):
        if not path.exists():
            raise FileNotFoundError(path)

    logger.info("Loading per-fold predictions from %s", args.pred_root)
    df_pred = load_all_folds(args.pred_root, args.tabpfn_dir, n_folds=args.n_folds)
    logger.info(
        "Loaded %d subject-fold rows across %d folds",
        len(df_pred), df_pred["fold"].nunique(),
    )

    logger.info("Loading metadata from %s", args.metadata_csv)
    meta = pd.read_csv(args.metadata_csv, low_memory=False)
    for col in _METADATA_COLS:
        if col not in meta.columns:
            raise KeyError(f"metadata.csv missing required column: {col!r}")
    df = df_pred.merge(meta[list(_METADATA_COLS)], on="ROSMAP_IndividualID", how="left")
    logger.info(
        "Metadata join coverage — apoe %d/%d, msex %d/%d, age %d/%d, "
        "educ %d/%d, cogdx %d/%d",
        df["apoe_genotype"].notna().sum(), len(df),
        df["msex"].notna().sum(), len(df),
        df["age_death"].notna().sum(), len(df),
        df["educ"].notna().sum(), len(df),
        df["cogdx"].notna().sum(), len(df),
    )

    df, stratum_meta = assign_strata(df)

    table = compute_subgroup_r2_table(df, n_folds=args.n_folds)

    # Overall (all-subjects) per-fold R² — useful as a reference baseline.
    canonical_per_fold: list[float] = []
    for fold in range(args.n_folds):
        sub = df[df["fold"] == fold]
        if len(sub) >= MIN_N_PER_FOLD_FOR_METRICS:
            canonical_per_fold.append(
                float(
                    r2_score(
                        sub["y_true"].to_numpy(dtype=np.float64),
                        sub["y_composite"].to_numpy(dtype=np.float64),
                    )
                )
            )
        else:
            canonical_per_fold.append(float("nan"))

    payload = {
        "config": {
            "pred_root": str(args.pred_root),
            "tabpfn_dir": str(args.tabpfn_dir),
            "metadata_csv": str(args.metadata_csv),
            "n_folds": int(args.n_folds),
            "min_n_per_fold_for_metrics": int(MIN_N_PER_FOLD_FOR_METRICS),
        },
        "stratum_definitions": stratum_meta,
        "overall_per_fold_r2": canonical_per_fold,
        "subgroup_table": table,
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with args.out_json.open("w") as fh:
        json.dump(payload, fh, indent=2, default=float)
    logger.info("Wrote %s", args.out_json)

    md = render_markdown_table(table, canonical_per_fold)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(md)
    logger.info("Wrote %s", args.out_md)

    fig_paths = render_forest_plot(table, canonical_per_fold, args.out_fig_dir)
    for fp in fig_paths:
        logger.info("Wrote %s", fp)

    _print_summary(table, canonical_per_fold)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
