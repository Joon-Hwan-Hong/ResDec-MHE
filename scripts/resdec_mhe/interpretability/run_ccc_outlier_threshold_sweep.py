#!/usr/bin/env python
"""CCC outlier threshold sensitivity sweep.

Companion to ``run_ccc_outlier_deepdive.py``. The deepdive uses the
baked-in 0.01 threshold inherited from
``per_subject_ccc_attention_summary.json`` (``n_high_attention_edges``).
This script repeats the outlier-vs-typical enrichment analysis at
multiple thresholds drawn from the RAW per-subject attention tensor
(``per_subject_ccc_attention.npz``) and emits a single comparative
JSON + MD + 4-panel figure so the reader can judge whether the
15-outlier signature is an artefact of the 0.01 cut or a genuine
threshold-stable signal.

Per threshold τ we compute, for the 516 subjects:

1. ``n_outliers``  — # subjects with at least one (CT, CT, edge_type)
   attention value ≥ τ.
2. ``cogn_global``  — Mann–Whitney U (two-sided) outliers vs typicals.
3. ``AD-dx``       — Fisher exact (two-sided) on cogdx ∈ {4,5}.
4. ``Sex``         — Fisher exact (two-sided) on msex==1.
5. ``Top-3 dominant edges``  — frequency of (source_ct → target_ct,
   edge_type) tuples in each subject's top-3 edges ≥ τ.

The 4-panel figure shows:
    A. n_outliers vs threshold
    B. cogn_global Mann–Whitney p vs threshold (log scale)
    C. AD-dx Fisher p vs threshold (log scale)
    D. Top-10 dominant-edge stability — Jaccard overlap of the top-10
       (src, tgt, edge_type) tuples between consecutive thresholds.

Outputs:
    outputs/canonical/interpretability/figures/ccc_heterogeneity/
        fig_threshold_sensitivity.{png,pdf}      (4-panel, 600 DPI)
    outputs/canonical/interpretability/ccc_heterogeneity/
        threshold_sensitivity.json
        threshold_sensitivity.md
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")  # must be set before pyplot import
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

_WORKTREE_ROOT = Path(__file__).resolve().parents[3]
if str(_WORKTREE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKTREE_ROOT))

from src.visualization.theme import apply_theme  # noqa: E402

logger = logging.getLogger(__name__)

# Cogdx semantics from ROSMAP: 1=NCI, 2=MCI, 3=MCI+, 4=AD-probable,
# 5=AD-possible, 6=Other dementia. Binarize AD = {4, 5}.
AD_COGDX_CODES = {4.0, 5.0}

DEFAULT_THRESHOLDS = (0.005, 0.01, 0.02, 0.05)
DEFAULT_TOP_K_PAIRS = 3
DEFAULT_TOP_N_EDGES_FOR_STABILITY = 10


def _parse_thresholds(s: str) -> list[float]:
    """Parse a comma-separated list of floats into a sorted ascending list."""
    out = [float(t.strip()) for t in s.split(",") if t.strip()]
    if not out:
        raise ValueError("Threshold list is empty.")
    return sorted(out)


def _per_subject_outlier_metrics(
    attention: np.ndarray, threshold: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(max_attention_above_tau, n_edges_above_tau)`` per subject.

    ``attention`` shape ``(n_subjects, n_ct, n_ct, n_edge_types)`` may
    contain NaN entries (≈81 % at canonical config); NaN is treated as
    missing — never counted toward ``n_edges`` or ``max_attention``.
    """
    n_subj = attention.shape[0]
    flat = attention.reshape(n_subj, -1)
    above = flat >= threshold  # NaN >= τ is False → safely excluded.
    n_edges = above.sum(axis=1).astype(int)

    # Max attention among those entries that exceed τ.  For subjects
    # with zero entries above τ, max_above is ``-inf`` from
    # ``np.where(above, flat, -inf).max(axis=1)``; we coerce it to NaN
    # so the JSON / MD report doesn't show a sentinel.
    masked = np.where(above, flat, -np.inf)
    max_above = masked.max(axis=1)
    max_above = np.where(np.isfinite(max_above), max_above, np.nan)
    return max_above, n_edges


def _top_k_edges_per_subject(
    attention: np.ndarray,
    cell_type_order: Sequence[str],
    edge_type_order: Sequence[str],
    threshold: float,
    top_k: int,
) -> list[list[tuple[str, str, str, float]]]:
    """For each subject, return the ``top_k`` edges with attention ≥ τ.

    Each edge is a tuple ``(source_ct, target_ct, edge_type, attention)``.
    Subjects with fewer than ``top_k`` qualifying edges return shorter
    lists. NaN entries are excluded (NaN >= τ is False).
    """
    n_subj, n_ct, _, n_et = attention.shape
    flat_dim = n_ct * n_ct * n_et
    out: list[list[tuple[str, str, str, float]]] = []
    for i in range(n_subj):
        flat = attention[i].reshape(flat_dim)
        # Mask NaN to a sentinel below threshold so argpartition skips them.
        finite = np.where(np.isfinite(flat), flat, -np.inf)
        # Indices of the ``top_k`` largest finite entries (descending).
        if top_k >= flat_dim:
            order = np.argsort(-finite)
        else:
            part = np.argpartition(-finite, top_k)[:top_k]
            order = part[np.argsort(-finite[part])]
        edges: list[tuple[str, str, str, float]] = []
        for idx in order:
            v = float(flat[idx])
            if not np.isfinite(v) or v < threshold:
                continue
            src_i, rest = divmod(int(idx), n_ct * n_et)
            tgt_i, et_i = divmod(rest, n_et)
            edges.append(
                (
                    str(cell_type_order[src_i]),
                    str(cell_type_order[tgt_i]),
                    str(edge_type_order[et_i]),
                    v,
                )
            )
        out.append(edges)
    return out


def _dominant_edge_counter(
    per_subject_top_edges: Iterable[list[tuple[str, str, str, float]]],
) -> Counter:
    """Count ``(source_ct, target_ct, edge_type)`` tuples across subjects.

    Each subject contributes at most one count per unique tuple — so
    the count is the number of subjects with that tuple in their
    top-K, not the total number of records.
    """
    c: Counter = Counter()
    for edges in per_subject_top_edges:
        seen = {(s, t, e) for (s, t, e, _v) in edges}
        for k in seen:
            c[k] += 1
    return c


def _join_metadata(
    subject_ids: Iterable[str],
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    """Inner-join subject_ids with ROSMAP metadata on ROSMAP_IndividualID."""
    sids = list(subject_ids)
    sub = metadata[metadata["ROSMAP_IndividualID"].isin(sids)].copy()
    sub = sub.drop_duplicates(subset=["ROSMAP_IndividualID"], keep="first")
    return sub


def _stats_outlier_vs_typical(
    out_meta: pd.DataFrame,
    typ_meta: pd.DataFrame,
) -> dict:
    """Mann–Whitney on cogn_global; Fisher exact on AD-dx + sex."""
    out_cog = out_meta["cogn_global"].dropna().to_numpy()
    typ_cog = typ_meta["cogn_global"].dropna().to_numpy()
    if out_cog.size and typ_cog.size:
        u_stat, p_cog = stats.mannwhitneyu(
            out_cog, typ_cog, alternative="two-sided"
        )
        u_stat = float(u_stat)
        p_cog = float(p_cog)
    else:
        u_stat = float("nan")
        p_cog = float("nan")

    def _ad_table(meta: pd.DataFrame) -> tuple[int, int]:
        cogdx = meta["cogdx"].dropna()
        ad = int(cogdx.isin(AD_COGDX_CODES).sum())
        non_ad = int((~cogdx.isin(AD_COGDX_CODES)).sum())
        return ad, non_ad

    o_ad, o_non = _ad_table(out_meta)
    t_ad, t_non = _ad_table(typ_meta)
    table = [[o_ad, o_non], [t_ad, t_non]]
    if any(sum(row) == 0 for row in table):
        odds_ratio, p_ad = float("nan"), float("nan")
    else:
        odds_ratio, p_ad = stats.fisher_exact(table, alternative="two-sided")
        odds_ratio = float(odds_ratio)
        p_ad = float(p_ad)

    def _sex_table(meta: pd.DataFrame) -> tuple[int, int]:
        sex = meta["msex"].dropna()
        male = int((sex == 1).sum())
        female = int((sex == 0).sum())
        return male, female

    o_m, o_f = _sex_table(out_meta)
    t_m, t_f = _sex_table(typ_meta)
    sex_tab = [[o_m, o_f], [t_m, t_f]]
    if any(sum(row) == 0 for row in sex_tab):
        sex_or, p_sex = float("nan"), float("nan")
    else:
        sex_or, p_sex = stats.fisher_exact(sex_tab, alternative="two-sided")
        sex_or = float(sex_or)
        p_sex = float(p_sex)

    return {
        "cogn_global": {
            "outlier_n": int(out_cog.size),
            "outlier_mean": float(out_cog.mean()) if out_cog.size else float("nan"),
            "outlier_median": (
                float(np.median(out_cog)) if out_cog.size else float("nan")
            ),
            "typical_n": int(typ_cog.size),
            "typical_mean": float(typ_cog.mean()) if typ_cog.size else float("nan"),
            "typical_median": (
                float(np.median(typ_cog)) if typ_cog.size else float("nan")
            ),
            "mannwhitney_u": u_stat,
            "mannwhitney_p_two_sided": p_cog,
        },
        "ad_dx_cogdx_4_or_5": {
            "outlier_ad": o_ad,
            "outlier_non_ad": o_non,
            "typical_ad": t_ad,
            "typical_non_ad": t_non,
            "outlier_ad_frac": (
                (o_ad / (o_ad + o_non)) if (o_ad + o_non) else float("nan")
            ),
            "typical_ad_frac": (
                (t_ad / (t_ad + t_non)) if (t_ad + t_non) else float("nan")
            ),
            "fisher_odds_ratio": odds_ratio,
            "fisher_p_two_sided": p_ad,
        },
        "sex_msex_male_eq_1": {
            "outlier_male": o_m,
            "outlier_female": o_f,
            "typical_male": t_m,
            "typical_female": t_f,
            "fisher_odds_ratio": sex_or,
            "fisher_p_two_sided": p_sex,
        },
    }


def _jaccard(a: Sequence[tuple[str, str, str]], b: Sequence[tuple[str, str, str]]) -> float:
    """|A ∩ B| / |A ∪ B|. Returns ``nan`` if both sets are empty."""
    sa = set(a)
    sb = set(b)
    union = sa | sb
    if not union:
        return float("nan")
    return float(len(sa & sb) / len(union))


def _compute_per_threshold(
    *,
    attention: np.ndarray,
    subject_ids: np.ndarray,
    folds: np.ndarray,
    cell_type_order: np.ndarray,
    edge_type_order: np.ndarray,
    metadata: pd.DataFrame,
    threshold: float,
    top_k_pairs: int,
    top_n_edges_for_stability: int,
) -> dict:
    """Compute outlier counts, enrichment, and dominant-edge tabulation."""
    max_above, n_edges = _per_subject_outlier_metrics(attention, threshold)
    is_outlier = n_edges > 0
    out_sids = subject_ids[is_outlier]
    typ_sids = subject_ids[~is_outlier]

    out_meta = _join_metadata(out_sids, metadata)
    typ_meta = _join_metadata(typ_sids, metadata)
    enrichment = _stats_outlier_vs_typical(out_meta, typ_meta)

    # Dominant-edge tabulation in OUTLIERS (top-K per subject).
    top_edges_outliers = _top_k_edges_per_subject(
        attention[is_outlier],
        cell_type_order=cell_type_order,
        edge_type_order=edge_type_order,
        threshold=threshold,
        top_k=top_k_pairs,
    )
    out_edge_counter = _dominant_edge_counter(top_edges_outliers)

    # Top-N (src, tgt, edge_type) for stability comparison + JSON dump.
    top_edges_sorted = sorted(
        out_edge_counter.items(), key=lambda kv: (-kv[1], kv[0])
    )
    top_n = top_edges_sorted[:top_n_edges_for_stability]
    top_n_keys = [k for k, _ in top_n]

    # Same tabulation in typicals (purely for the JSON / MD context).
    top_edges_typicals = _top_k_edges_per_subject(
        attention[~is_outlier],
        cell_type_order=cell_type_order,
        edge_type_order=edge_type_order,
        threshold=threshold,
        top_k=top_k_pairs,
    )
    typ_edge_counter = _dominant_edge_counter(top_edges_typicals)
    typ_top_n = sorted(typ_edge_counter.items(), key=lambda kv: (-kv[1], kv[0]))[
        :top_n_edges_for_stability
    ]

    # Per-outlier records (for JSON drilldown). Sort by max_above desc.
    outlier_records: list[dict] = []
    out_idx = np.flatnonzero(is_outlier)
    for local_i, global_i in enumerate(out_idx):
        sid = str(subject_ids[global_i])
        row = out_meta[out_meta["ROSMAP_IndividualID"] == sid].head(1)
        rec: dict = {
            "subject_id": sid,
            "fold": int(folds[global_i]),
            "max_attention_above_tau": (
                float(max_above[global_i])
                if np.isfinite(max_above[global_i])
                else float("nan")
            ),
            "n_edges_above_tau": int(n_edges[global_i]),
            "top_edges": [
                {
                    "source_ct": s,
                    "target_ct": t,
                    "edge_type": e,
                    "attention": v,
                }
                for (s, t, e, v) in top_edges_outliers[local_i]
            ],
        }
        for c in ("cogn_global", "cogdx", "msex", "age_bl", "educ"):
            if not row.empty and not pd.isna(row[c].iloc[0]):
                rec[c] = float(row[c].iloc[0])
            else:
                rec[c] = float("nan")
        outlier_records.append(rec)
    outlier_records.sort(
        key=lambda r: (
            -r["max_attention_above_tau"]
            if np.isfinite(r["max_attention_above_tau"])
            else 0.0
        )
    )

    return {
        "threshold": float(threshold),
        "n_outliers": int(is_outlier.sum()),
        "n_typical": int((~is_outlier).sum()),
        "enrichment": enrichment,
        "top_n_dominant_edges_outliers": [
            {
                "source_ct": k[0],
                "target_ct": k[1],
                "edge_type": k[2],
                "n_outlier_subjects": int(v),
            }
            for k, v in top_n
        ],
        "top_n_dominant_edges_typicals": [
            {
                "source_ct": k[0],
                "target_ct": k[1],
                "edge_type": k[2],
                "n_typical_subjects": int(v),
            }
            for k, v in typ_top_n
        ],
        "_top_n_keys": top_n_keys,  # internal: passed to stability metric
        "outlier_subjects": outlier_records,
    }


def _render_figure(
    *,
    per_thr: list[dict],
    out_path_png: Path,
    out_path_pdf: Path,
) -> None:
    apply_theme(style="paper", use_scienceplots=True)
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 10.0))
    thrs = np.asarray([r["threshold"] for r in per_thr], dtype=float)

    # ---- Panel A: n_outliers vs threshold ----
    ax = axes[0, 0]
    n_out = np.asarray([r["n_outliers"] for r in per_thr], dtype=int)
    ax.plot(thrs, n_out, marker="o", color="#d62728", linewidth=1.5)
    for x, y in zip(thrs, n_out):
        ax.annotate(
            f"n={y}",
            (x, y),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=8,
        )
    ax.set_xscale("log")
    ax.set_xlabel(r"Threshold $\tau$ (attention)", fontsize=11)
    ax.set_ylabel("# outlier subjects", fontsize=11)
    ax.set_title("(a) Outlier count vs threshold", fontsize=12)
    ax.grid(linestyle=":", alpha=0.4)

    # ---- Panel B: cogn_global Mann-Whitney p vs threshold ----
    ax = axes[0, 1]
    p_cog = np.asarray(
        [r["enrichment"]["cogn_global"]["mannwhitney_p_two_sided"] for r in per_thr],
        dtype=float,
    )
    ax.plot(thrs, p_cog, marker="o", color="#1f77b4", linewidth=1.5)
    for x, y in zip(thrs, p_cog):
        if np.isfinite(y):
            ax.annotate(
                f"p={y:.3g}",
                (x, y),
                textcoords="offset points",
                xytext=(6, 6),
                fontsize=8,
            )
    ax.axhline(0.05, color="grey", linestyle="--", linewidth=0.8, label=r"$\alpha=0.05$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Threshold $\tau$", fontsize=11)
    ax.set_ylabel("cogn_global Mann–Whitney p (two-sided)", fontsize=11)
    ax.set_title("(b) cogn_global p vs threshold", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(linestyle=":", alpha=0.4)

    # ---- Panel C: AD-dx Fisher p vs threshold ----
    ax = axes[1, 0]
    p_ad = np.asarray(
        [r["enrichment"]["ad_dx_cogdx_4_or_5"]["fisher_p_two_sided"] for r in per_thr],
        dtype=float,
    )
    ax.plot(thrs, p_ad, marker="o", color="#2ca02c", linewidth=1.5)
    for x, y in zip(thrs, p_ad):
        if np.isfinite(y):
            ax.annotate(
                f"p={y:.3g}",
                (x, y),
                textcoords="offset points",
                xytext=(6, 6),
                fontsize=8,
            )
    ax.axhline(0.05, color="grey", linestyle="--", linewidth=0.8, label=r"$\alpha=0.05$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Threshold $\tau$", fontsize=11)
    ax.set_ylabel("AD-dx Fisher p (two-sided)", fontsize=11)
    ax.set_title("(c) AD-dx p vs threshold", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(linestyle=":", alpha=0.4)

    # ---- Panel D: top-10 dominant-edge stability between consecutive τ ----
    ax = axes[1, 1]
    if len(per_thr) < 2:
        ax.text(
            0.5,
            0.5,
            "Need ≥ 2 thresholds for stability",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=11,
        )
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        labels: list[str] = []
        jaccards: list[float] = []
        intersect_counts: list[int] = []
        for i in range(1, len(per_thr)):
            prev = per_thr[i - 1]["_top_n_keys"]
            cur = per_thr[i]["_top_n_keys"]
            j = _jaccard(prev, cur)
            inter = len(set(prev) & set(cur))
            labels.append(
                f"{per_thr[i-1]['threshold']:g} → {per_thr[i]['threshold']:g}"
            )
            jaccards.append(j)
            intersect_counts.append(inter)
        x = np.arange(len(labels))
        bars = ax.bar(
            x,
            jaccards,
            color="#9467bd",
            alpha=0.78,
            edgecolor="black",
            linewidth=0.8,
        )
        for b, j, ic in zip(bars, jaccards, intersect_counts):
            label = (
                f"{j:.2f}\n|∩|={ic}"
                if np.isfinite(j)
                else f"NA\n|∩|={ic}"
            )
            y0 = (j if np.isfinite(j) else 0.0) + 0.02
            ax.text(
                b.get_x() + b.get_width() / 2.0,
                y0,
                label,
                ha="center",
                va="bottom",
                fontsize=8,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
        ax.set_ylim(0.0, 1.1)
        ax.set_ylabel("Jaccard overlap (top-10)", fontsize=11)
        ax.set_title(
            "(d) Dominant-edge top-10 stability (consecutive τ)", fontsize=12
        )
        ax.grid(axis="y", linestyle=":", alpha=0.4)

    fig.suptitle(
        "CCC outlier threshold sensitivity — n=516 subjects",
        fontsize=13,
        y=1.0,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    out_path_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path_png, dpi=600, bbox_inches="tight")
    fig.savefig(out_path_pdf, dpi=600, bbox_inches="tight")
    plt.close(fig)


def _build_markdown(
    *,
    per_thr: list[dict],
    n_subjects: int,
    n_cell_types: int,
    n_edge_types: int,
    top_k_pairs: int,
    top_n_edges_for_stability: int,
    raw_npz: Path,
    metadata_csv: Path,
) -> str:
    lines: list[str] = []
    lines.append("# CCC Outlier Threshold Sensitivity Sweep")
    lines.append("")
    lines.append(
        f"- Source tensor: `{raw_npz.relative_to(_WORKTREE_ROOT)}` "
        f"({n_subjects} subjects × {n_cell_types}² CT × "
        f"{n_edge_types} edge types)"
    )
    lines.append(f"- Metadata: `{metadata_csv.relative_to(_WORKTREE_ROOT)}`")
    lines.append(
        f"- Per-subject top-K used for dominant-edge tabulation: "
        f"K = {top_k_pairs}; stability over top-{top_n_edges_for_stability}"
    )
    lines.append("")
    lines.append("## 1. Comparative table")
    lines.append("")
    lines.append(
        "| τ | n_outliers | n_typical | "
        "cogn_global p | AD-dx p | Sex p | "
        "AD-dx OR | Sex OR | top-10 Jaccard vs prev τ |"
    )
    lines.append(
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"
    )
    for i, r in enumerate(per_thr):
        cog = r["enrichment"]["cogn_global"]
        ad = r["enrichment"]["ad_dx_cogdx_4_or_5"]
        sx = r["enrichment"]["sex_msex_male_eq_1"]
        if i == 0:
            jstr = "—"
        else:
            jstr = f"{_jaccard(per_thr[i-1]['_top_n_keys'], r['_top_n_keys']):.3f}"
        lines.append(
            f"| {r['threshold']:g} | {r['n_outliers']} | {r['n_typical']} | "
            f"{cog['mannwhitney_p_two_sided']:.4g} | "
            f"{ad['fisher_p_two_sided']:.4g} | "
            f"{sx['fisher_p_two_sided']:.4g} | "
            f"{ad['fisher_odds_ratio']:.3g} | "
            f"{sx['fisher_odds_ratio']:.3g} | "
            f"{jstr} |"
        )
    lines.append("")
    lines.append("## 2. Per-threshold details")
    lines.append("")
    for r in per_thr:
        lines.append(f"### τ = {r['threshold']:g}")
        lines.append("")
        cog = r["enrichment"]["cogn_global"]
        ad = r["enrichment"]["ad_dx_cogdx_4_or_5"]
        sx = r["enrichment"]["sex_msex_male_eq_1"]
        lines.append(
            f"- **Outliers:** {r['n_outliers']} (typical: {r['n_typical']})"
        )
        lines.append(
            f"- **cogn_global** outlier mean = {cog['outlier_mean']:.3f} "
            f"(median {cog['outlier_median']:.3f}, n={cog['outlier_n']}); "
            f"typical mean = {cog['typical_mean']:.3f} "
            f"(median {cog['typical_median']:.3f}, n={cog['typical_n']}). "
            f"Mann–Whitney U = {cog['mannwhitney_u']:.1f}, "
            f"p = **{cog['mannwhitney_p_two_sided']:.4g}**"
        )
        lines.append(
            f"- **AD-dx (cogdx ∈ {{4,5}})** outlier "
            f"{ad['outlier_ad']}/{ad['outlier_ad'] + ad['outlier_non_ad']} "
            f"({ad['outlier_ad_frac']:.2%}); typical "
            f"{ad['typical_ad']}/{ad['typical_ad'] + ad['typical_non_ad']} "
            f"({ad['typical_ad_frac']:.2%}). "
            f"Fisher OR = {ad['fisher_odds_ratio']:.3g}, "
            f"p = **{ad['fisher_p_two_sided']:.4g}**"
        )
        lines.append(
            f"- **Sex (msex=1 male)** outlier "
            f"M={sx['outlier_male']} F={sx['outlier_female']}; typical "
            f"M={sx['typical_male']} F={sx['typical_female']}. "
            f"Fisher OR = {sx['fisher_odds_ratio']:.3g}, "
            f"p = **{sx['fisher_p_two_sided']:.4g}**"
        )
        lines.append("")
        lines.append(
            f"#### Top-{top_n_edges_for_stability} dominant edges (outliers, top-{top_k_pairs} per subject)"
        )
        lines.append("")
        lines.append("| Source CT | Target CT | Edge type | # outlier subjects |")
        lines.append("| --- | --- | --- | --- |")
        for e in r["top_n_dominant_edges_outliers"]:
            lines.append(
                f"| {e['source_ct']} | {e['target_ct']} | {e['edge_type']} | "
                f"{e['n_outlier_subjects']} |"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ccc-npz",
        type=Path,
        default=_WORKTREE_ROOT
        / "outputs/canonical/interpretability/ccc/per_subject_ccc_attention.npz",
        help="Path to per_subject_ccc_attention.npz (raw tensor).",
    )
    parser.add_argument(
        "--metadata-csv",
        type=Path,
        default=_WORKTREE_ROOT / "data/metadata_ROSMAP/metadata.csv",
        help="Path to ROSMAP metadata.csv.",
    )
    parser.add_argument(
        "--out-fig-dir",
        type=Path,
        default=_WORKTREE_ROOT
        / "outputs/canonical/interpretability/figures/ccc_heterogeneity",
        help="Output directory for the figure files.",
    )
    parser.add_argument(
        "--out-data-dir",
        type=Path,
        default=_WORKTREE_ROOT
        / "outputs/canonical/interpretability/ccc_heterogeneity",
        help="Output directory for the JSON + MD files.",
    )
    parser.add_argument(
        "--thresholds",
        type=_parse_thresholds,
        default=list(DEFAULT_THRESHOLDS),
        help="Comma-separated thresholds (default: 0.005,0.01,0.02,0.05).",
    )
    parser.add_argument(
        "--top-k-pairs",
        type=int,
        default=DEFAULT_TOP_K_PAIRS,
        help="Top-K edges per subject used for pair tabulation.",
    )
    parser.add_argument(
        "--top-n-edges-for-stability",
        type=int,
        default=DEFAULT_TOP_N_EDGES_FOR_STABILITY,
        help="Top-N dominant edges used for cross-threshold Jaccard stability.",
    )
    args = parser.parse_args()

    if not args.ccc_npz.is_file():
        logger.error("CCC NPZ not found: %s", args.ccc_npz)
        return 1
    if not args.metadata_csv.is_file():
        logger.error("Metadata CSV not found: %s", args.metadata_csv)
        return 1

    npz = np.load(args.ccc_npz)
    attention = np.asarray(npz["attention"], dtype=np.float32)
    subject_ids = np.asarray(npz["subject_ids"]).astype(str)
    folds = np.asarray(npz["folds"], dtype=int)
    cell_type_order = np.asarray(npz["cell_type_order"]).astype(str)
    edge_type_order = np.asarray(npz["edge_type_order"]).astype(str)
    n_subj, n_ct, n_ct_b, n_et = attention.shape
    if n_ct != n_ct_b or n_ct != cell_type_order.size or n_et != edge_type_order.size:
        logger.error(
            "Tensor shape mismatch: attention=%s, ct_order=%d, et_order=%d",
            attention.shape,
            cell_type_order.size,
            edge_type_order.size,
        )
        return 1
    logger.info(
        "Loaded attention tensor: %d subjects × %d² CT × %d edge types "
        "(NaN frac = %.3f)",
        n_subj,
        n_ct,
        n_et,
        float(np.isnan(attention).sum() / attention.size),
    )

    metadata = pd.read_csv(args.metadata_csv)

    per_thr: list[dict] = []
    for tau in args.thresholds:
        rec = _compute_per_threshold(
            attention=attention,
            subject_ids=subject_ids,
            folds=folds,
            cell_type_order=cell_type_order,
            edge_type_order=edge_type_order,
            metadata=metadata,
            threshold=float(tau),
            top_k_pairs=args.top_k_pairs,
            top_n_edges_for_stability=args.top_n_edges_for_stability,
        )
        logger.info(
            "τ=%g: n_outliers=%d, cogn_global p=%.4g, AD-dx p=%.4g",
            tau,
            rec["n_outliers"],
            rec["enrichment"]["cogn_global"]["mannwhitney_p_two_sided"],
            rec["enrichment"]["ad_dx_cogdx_4_or_5"]["fisher_p_two_sided"],
        )
        per_thr.append(rec)

    # --- Figure ---
    fig_png = args.out_fig_dir / "fig_threshold_sensitivity.png"
    fig_pdf = args.out_fig_dir / "fig_threshold_sensitivity.pdf"
    _render_figure(per_thr=per_thr, out_path_png=fig_png, out_path_pdf=fig_pdf)
    logger.info("Wrote %s", fig_png)
    logger.info("Wrote %s", fig_pdf)

    # --- JSON ---
    args.out_data_dir.mkdir(parents=True, exist_ok=True)

    # Stability metric across consecutive thresholds.
    stability: list[dict] = []
    for i in range(1, len(per_thr)):
        prev = per_thr[i - 1]
        cur = per_thr[i]
        prev_keys = prev["_top_n_keys"]
        cur_keys = cur["_top_n_keys"]
        stability.append(
            {
                "threshold_prev": prev["threshold"],
                "threshold_cur": cur["threshold"],
                "jaccard_top_n": _jaccard(prev_keys, cur_keys),
                "intersection_size": int(len(set(prev_keys) & set(cur_keys))),
                "union_size": int(len(set(prev_keys) | set(cur_keys))),
                "top_n": int(args.top_n_edges_for_stability),
            }
        )

    payload = {
        "config": {
            "thresholds": [float(t) for t in args.thresholds],
            "top_k_pairs": int(args.top_k_pairs),
            "top_n_edges_for_stability": int(args.top_n_edges_for_stability),
            "n_subjects": int(n_subj),
            "n_cell_types": int(n_ct),
            "n_edge_types": int(n_et),
            "ccc_npz": str(args.ccc_npz),
            "metadata_csv": str(args.metadata_csv),
        },
        "per_threshold": [
            {k: v for k, v in r.items() if not k.startswith("_")} for r in per_thr
        ],
        "stability_consecutive": stability,
    }
    json_path = args.out_data_dir / "threshold_sensitivity.json"
    json_path.write_text(json.dumps(payload, indent=2))
    logger.info("Wrote %s", json_path)

    # --- Markdown ---
    md = _build_markdown(
        per_thr=per_thr,
        n_subjects=int(n_subj),
        n_cell_types=int(n_ct),
        n_edge_types=int(n_et),
        top_k_pairs=int(args.top_k_pairs),
        top_n_edges_for_stability=int(args.top_n_edges_for_stability),
        raw_npz=args.ccc_npz,
        metadata_csv=args.metadata_csv,
    )
    md_path = args.out_data_dir / "threshold_sensitivity.md"
    md_path.write_text(md)
    logger.info("Wrote %s", md_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
