#!/usr/bin/env python
"""Per-subject CCC heterogeneity deepdive — outlier vs typical.

Loads the per-subject CCC attention summary
(``per_subject_ccc_attention_summary.json``) plus subject metadata
(``data/metadata_ROSMAP/metadata.csv``), then:

1. Identifies the 15 outlier subjects (``n_high_attention_edges > 0``)
   and 501 typical subjects.
2. Joins on ``ROSMAP_IndividualID`` and tests for enrichment of the
   outliers vs typical subjects on:
       - cogn_global  (Mann–Whitney U / Wilcoxon rank-sum)
       - AD-dx (cogdx; binarized AD = cogdx in {4, 5} vs other) (Fisher exact)
3. Tabulates the dominant edges (>0.01 attention) of each outlier and
   reports the most-frequent CT pairs (top-3 per subject) inside the
   outlier vs typical subsets.

Outputs:
    outputs/canonical/interpretability/figures/ccc_heterogeneity/
        fig_ccc_outlier_demographics.{png,pdf}    (4-panel, 600 DPI)
    outputs/canonical/interpretability/ccc_heterogeneity/
        per_subject_outlier_analysis.json
        per_subject_outlier_analysis.md

Caveat: the URL (Upper-rhombic-lip) cell type is sparse
(zero_frac ≈ 0.79), so any URL involvement in outlier edges is
coverage-artifact-affected; this is logged in both JSON and MD.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable

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
# 5=AD-possible, 6=Other dementia. We binarize AD = {4, 5} per the
# ROSMAP convention used elsewhere in this repo.
AD_COGDX_CODES = {4.0, 5.0}


def _split_outlier_typical(
    summary: dict,
) -> tuple[list[dict], list[dict]]:
    """Return (outliers, typicals) by ``n_high_attention_edges > 0``."""
    per_sub = summary["per_subject"]
    outliers = [s for s in per_sub if s["n_high_attention_edges"] > 0]
    typicals = [s for s in per_sub if s["n_high_attention_edges"] == 0]
    return outliers, typicals


def _join_metadata(
    subject_records: Iterable[dict],
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    """Inner-join subject records with ROSMAP metadata on ROSMAP_IndividualID."""
    sids = [s["subject_id"] for s in subject_records]
    sub = metadata[metadata["ROSMAP_IndividualID"].isin(sids)].copy()
    sub = sub.drop_duplicates(subset=["ROSMAP_IndividualID"], keep="first")
    return sub


def _stats_outlier_vs_typical(
    out_meta: pd.DataFrame,
    typ_meta: pd.DataFrame,
) -> dict:
    """Compute Wilcoxon rank-sum on cogn_global; Fisher exact on AD-dx."""
    out_cog = out_meta["cogn_global"].dropna().to_numpy()
    typ_cog = typ_meta["cogn_global"].dropna().to_numpy()
    if out_cog.size and typ_cog.size:
        u_stat, p_cog = stats.mannwhitneyu(out_cog, typ_cog, alternative="two-sided")
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

    # Sex (msex 1=male, 0=female) — exploratory Fisher exact.
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
            "outlier_median": float(np.median(out_cog)) if out_cog.size else float("nan"),
            "typical_n": int(typ_cog.size),
            "typical_mean": float(typ_cog.mean()) if typ_cog.size else float("nan"),
            "typical_median": float(np.median(typ_cog)) if typ_cog.size else float("nan"),
            "mannwhitney_u": u_stat,
            "mannwhitney_p_two_sided": p_cog,
            "test": "Mann–Whitney U (two-sided), surrogate for Wilcoxon rank-sum",
        },
        "ad_dx_cogdx_4_or_5": {
            "outlier_ad": o_ad,
            "outlier_non_ad": o_non,
            "typical_ad": t_ad,
            "typical_non_ad": t_non,
            "outlier_ad_frac": (o_ad / (o_ad + o_non)) if (o_ad + o_non) else float("nan"),
            "typical_ad_frac": (t_ad / (t_ad + t_non)) if (t_ad + t_non) else float("nan"),
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


def _dominant_edges_per_subject(
    subject_records: Iterable[dict],
    threshold: float,
) -> dict[str, list[dict]]:
    """Return ``{subject_id: [edges with attention >= threshold (top_edges only)]}``."""
    out: dict[str, list[dict]] = {}
    for s in subject_records:
        sid = s["subject_id"]
        kept = [e for e in s.get("top_edges", []) if e.get("attention", 0.0) >= threshold]
        out[sid] = kept
    return out


def _top_pair_frequency(
    subject_records: Iterable[dict],
    *,
    top_k: int = 3,
) -> Counter:
    """Frequency of (source_ct -> target_ct) in each subject's top-K edges."""
    c: Counter = Counter()
    for s in subject_records:
        for e in s.get("top_edges", [])[:top_k]:
            key = (e["source_ct"], e["target_ct"])
            c[key] += 1
    return c


def _pair_heatmap_matrix(
    pair_counter: Counter,
    cell_types: list[str],
) -> np.ndarray:
    """N x N matrix of pair-frequency counts; rows=source, cols=target."""
    idx = {ct: i for i, ct in enumerate(cell_types)}
    mat = np.zeros((len(cell_types), len(cell_types)), dtype=int)
    for (src, tgt), cnt in pair_counter.items():
        if src in idx and tgt in idx:
            mat[idx[src], idx[tgt]] = cnt
    return mat


def _render_figure(
    *,
    out_meta: pd.DataFrame,
    typ_meta: pd.DataFrame,
    out_pairs: Counter,
    typ_pairs: Counter,
    cell_types: list[str],
    out_path_png: Path,
    out_path_pdf: Path,
) -> None:
    apply_theme(style="paper", use_scienceplots=True)
    fig, axes = plt.subplots(2, 2, figsize=(14.0, 12.0))

    # ---- Panel A: cogn_global distribution outliers vs typical ----
    ax = axes[0, 0]
    out_cog = out_meta["cogn_global"].dropna().to_numpy()
    typ_cog = typ_meta["cogn_global"].dropna().to_numpy()
    bins = np.linspace(min(out_cog.min(), typ_cog.min()) - 0.1,
                       max(out_cog.max(), typ_cog.max()) + 0.1, 30)
    ax.hist(typ_cog, bins=bins, color="#1f77b4", alpha=0.55,
            label=f"Typical (n={typ_cog.size})", density=True)
    ax.hist(out_cog, bins=bins, color="#d62728", alpha=0.75,
            label=f"Outliers (n={out_cog.size})", density=True)
    ax.axvline(np.median(typ_cog), color="#1f77b4", linestyle=":", linewidth=1.2)
    ax.axvline(np.median(out_cog), color="#d62728", linestyle=":", linewidth=1.2)
    ax.set_xlabel("cogn_global", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("(a) cogn_global — outliers vs typical", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(linestyle=":", alpha=0.4)

    # ---- Panel B: AD-dx fraction (cogdx in {4,5}) ----
    ax = axes[0, 1]
    o_ad = int(out_meta["cogdx"].dropna().isin(AD_COGDX_CODES).sum())
    o_total = int(out_meta["cogdx"].dropna().shape[0])
    t_ad = int(typ_meta["cogdx"].dropna().isin(AD_COGDX_CODES).sum())
    t_total = int(typ_meta["cogdx"].dropna().shape[0])
    fracs = [
        (o_ad / o_total) if o_total else 0.0,
        (t_ad / t_total) if t_total else 0.0,
    ]
    labels = [f"Outliers ({o_ad}/{o_total})", f"Typical ({t_ad}/{t_total})"]
    bars = ax.bar(labels, fracs, color=["#d62728", "#1f77b4"], alpha=0.75,
                  edgecolor="black", linewidth=1.0)
    for b, frac in zip(bars, fracs):
        ax.text(b.get_x() + b.get_width() / 2.0, frac + 0.02,
                f"{frac:.2%}", ha="center", fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Fraction AD-dx (cogdx ∈ {4, 5})", fontsize=11)
    ax.set_title("(b) AD-dx fraction — outliers vs typical", fontsize=12)
    ax.grid(axis="y", linestyle=":", alpha=0.4)

    # ---- Helper for a top-CT-pairs heatmap (panels C, D) ----
    def _draw_heatmap(ax_, counter: Counter, cell_types_: list[str], title_: str) -> None:
        # Identify the union of cell types appearing as source or target
        # in this counter, ranked by total degree.
        deg = Counter()
        for (src, tgt), cnt in counter.items():
            deg[src] += cnt
            deg[tgt] += cnt
        kept = [ct for ct, _ in deg.most_common(12)]
        kept = [ct for ct in cell_types_ if ct in kept]  # preserve canonical order
        if not kept:
            ax_.text(0.5, 0.5, "No data", transform=ax_.transAxes,
                     ha="center", va="center", fontsize=11)
            ax_.set_title(title_, fontsize=12)
            ax_.set_xticks([])
            ax_.set_yticks([])
            return
        idx = {ct: i for i, ct in enumerate(kept)}
        mat = np.zeros((len(kept), len(kept)), dtype=int)
        for (src, tgt), cnt in counter.items():
            if src in idx and tgt in idx:
                mat[idx[src], idx[tgt]] = cnt
        im = ax_.imshow(mat, cmap="Reds", aspect="auto")
        ax_.set_xticks(np.arange(len(kept)))
        ax_.set_yticks(np.arange(len(kept)))
        ax_.set_xticklabels(kept, rotation=70, ha="right", fontsize=8)
        ax_.set_yticklabels(kept, fontsize=8)
        ax_.set_title(title_, fontsize=12)
        ax_.set_xlabel("target CT", fontsize=10)
        ax_.set_ylabel("source CT", fontsize=10)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if mat[i, j] > 0:
                    color = "white" if mat[i, j] > mat.max() / 2.0 else "black"
                    ax_.text(j, i, int(mat[i, j]), ha="center", va="center",
                             fontsize=7, color=color)
        plt.colorbar(im, ax=ax_, fraction=0.046, pad=0.04, label="# subjects")

    _draw_heatmap(
        axes[1, 0],
        out_pairs,
        cell_types,
        f"(c) Top-3 CT pair frequency — outliers (n={int(out_meta.shape[0])})",
    )
    _draw_heatmap(
        axes[1, 1],
        typ_pairs,
        cell_types,
        f"(d) Top-3 CT pair frequency — typical (n={int(typ_meta.shape[0])})",
    )

    fig.suptitle(
        "CCC heterogeneity — 15 outlier subjects vs 501 typical subjects "
        "(threshold = 0.01)",
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
    n_outliers: int,
    n_typical: int,
    threshold: float,
    enrichment: dict,
    outlier_table: pd.DataFrame,
    out_top_pairs: list[tuple[tuple[str, str], int]],
    typ_top_pairs: list[tuple[tuple[str, str], int]],
    url_caveat: bool,
) -> str:
    lines: list[str] = []
    lines.append("# CCC Heterogeneity Deepdive — Outlier vs Typical")
    lines.append("")
    lines.append(
        f"- Threshold for high attention: **{threshold}**"
        f" — {n_outliers} outliers vs {n_typical} typical subjects"
    )
    lines.append("")
    lines.append("## 1. Enrichment of clinical phenotype")
    cog = enrichment["cogn_global"]
    lines.append(
        f"- **cogn_global** outliers (n={cog['outlier_n']}): mean={cog['outlier_mean']:.3f}, "
        f"median={cog['outlier_median']:.3f}; typical (n={cog['typical_n']}): "
        f"mean={cog['typical_mean']:.3f}, median={cog['typical_median']:.3f}"
    )
    lines.append(
        f"  - Mann–Whitney U={cog['mannwhitney_u']:.1f}, "
        f"two-sided p = **{cog['mannwhitney_p_two_sided']:.4g}**"
    )
    ad = enrichment["ad_dx_cogdx_4_or_5"]
    lines.append(
        f"- **AD-dx (cogdx ∈ {{4,5}})** outliers: {ad['outlier_ad']}/{ad['outlier_ad'] + ad['outlier_non_ad']} "
        f"({ad['outlier_ad_frac']:.2%}); "
        f"typical: {ad['typical_ad']}/{ad['typical_ad'] + ad['typical_non_ad']} "
        f"({ad['typical_ad_frac']:.2%})"
    )
    lines.append(
        f"  - Fisher exact OR={ad['fisher_odds_ratio']:.3g}, "
        f"two-sided p = **{ad['fisher_p_two_sided']:.4g}**"
    )
    sx = enrichment["sex_msex_male_eq_1"]
    lines.append(
        f"- **Sex (msex=1 male)** outliers: M={sx['outlier_male']} F={sx['outlier_female']}; "
        f"typical: M={sx['typical_male']} F={sx['typical_female']}"
    )
    lines.append(
        f"  - Fisher exact OR={sx['fisher_odds_ratio']:.3g}, "
        f"two-sided p = **{sx['fisher_p_two_sided']:.4g}**"
    )
    lines.append("")
    lines.append("## 2. Top CT pairs (top-3 per subject, frequency over the 15 outliers)")
    lines.append("")
    lines.append("| Source CT | Target CT | # subjects |")
    lines.append("| --- | --- | --- |")
    for (src, tgt), cnt in out_top_pairs:
        lines.append(f"| {src} | {tgt} | {cnt} |")
    lines.append("")
    lines.append("## 3. Top CT pairs (top-3 per subject, frequency over the 501 typical)")
    lines.append("")
    lines.append("| Source CT | Target CT | # subjects |")
    lines.append("| --- | --- | --- |")
    for (src, tgt), cnt in typ_top_pairs:
        lines.append(f"| {src} | {tgt} | {cnt} |")
    lines.append("")
    lines.append("## 4. Per-outlier table")
    lines.append("")
    lines.append("| subject_id | fold | max_attention | n_high_edges | cogn_global | cogdx | msex | age_bl | educ |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for _, row in outlier_table.iterrows():
        lines.append(
            "| {sid} | {fold} | {ma:.4f} | {n_he} | {cg} | {cd} | {sx} | {ab} | {ed} |".format(
                sid=row["subject_id"],
                fold=row["fold"],
                ma=float(row["max_attention"]),
                n_he=int(row["n_high_attention_edges"]),
                cg=("nan" if pd.isna(row.get("cogn_global"))
                    else f"{float(row['cogn_global']):.3f}"),
                cd=("nan" if pd.isna(row.get("cogdx"))
                    else f"{float(row['cogdx']):.0f}"),
                sx=("nan" if pd.isna(row.get("msex"))
                    else f"{int(row['msex'])}"),
                ab=("nan" if pd.isna(row.get("age_bl"))
                    else f"{float(row['age_bl']):.1f}"),
                ed=("nan" if pd.isna(row.get("educ"))
                    else f"{int(row['educ'])}"),
            )
        )
    lines.append("")
    if url_caveat:
        lines.append(
            "> **Caveat (URL coverage):** the URL (Upper-rhombic-lip) cell type "
            "is sparse (zero_frac ≈ 0.79 across subjects), so any URL "
            "involvement in the outlier dominant edges should be treated as "
            "coverage-artifact-affected, not biology-bearing."
        )
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ccc-summary-json",
        type=Path,
        default=_WORKTREE_ROOT
        / "outputs/canonical/interpretability/ccc/per_subject_ccc_attention_summary.json",
        help="Path to per_subject_ccc_attention_summary.json.",
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
        "--threshold",
        type=float,
        default=0.01,
        help="High-attention threshold (default: 0.01, matches summary).",
    )
    parser.add_argument(
        "--top-k-pairs",
        type=int,
        default=3,
        help="Top-K edges per subject used for pair-frequency tabulation.",
    )
    args = parser.parse_args()

    if not args.ccc_summary_json.is_file():
        logger.error("CCC summary JSON not found: %s", args.ccc_summary_json)
        return 1
    if not args.metadata_csv.is_file():
        logger.error("Metadata CSV not found: %s", args.metadata_csv)
        return 1

    summary = json.loads(args.ccc_summary_json.read_text())
    metadata = pd.read_csv(args.metadata_csv)

    outliers, typicals = _split_outlier_typical(summary)
    logger.info("Outliers: n=%d ; Typical: n=%d", len(outliers), len(typicals))

    out_meta = _join_metadata(outliers, metadata)
    typ_meta = _join_metadata(typicals, metadata)

    enrichment = _stats_outlier_vs_typical(out_meta, typ_meta)
    logger.info(
        "cogn_global Mann–Whitney p=%.4g ; AD-dx Fisher p=%.4g",
        enrichment["cogn_global"]["mannwhitney_p_two_sided"],
        enrichment["ad_dx_cogdx_4_or_5"]["fisher_p_two_sided"],
    )

    dominant = _dominant_edges_per_subject(outliers, threshold=args.threshold)
    out_pairs = _top_pair_frequency(outliers, top_k=args.top_k_pairs)
    typ_pairs = _top_pair_frequency(typicals, top_k=args.top_k_pairs)

    cell_types = sorted(
        {e["source_ct"] for s in summary["per_subject"] for e in s.get("top_edges", [])}
        | {e["target_ct"] for s in summary["per_subject"] for e in s.get("top_edges", [])}
    )

    # URL caveat trigger: any URL appearance in outlier top edges?
    url_in_outliers = any(
        ("URL" in e.get("source_ct", "") or "URL" in e.get("target_ct", ""))
        for s in outliers
        for e in s.get("top_edges", [])
    )

    # Per-outlier joined table for the MD report (sorted for stability).
    outlier_records = []
    for s in outliers:
        sid = s["subject_id"]
        row = out_meta[out_meta["ROSMAP_IndividualID"] == sid].head(1)
        rec = {
            "subject_id": sid,
            "fold": s["fold"],
            "max_attention": s["max_attention"],
            "n_high_attention_edges": s["n_high_attention_edges"],
        }
        for c in ("cogn_global", "cogdx", "msex", "age_bl", "educ"):
            rec[c] = float(row[c].iloc[0]) if not row.empty and not pd.isna(row[c].iloc[0]) else float("nan")
        outlier_records.append(rec)
    outlier_df = pd.DataFrame(outlier_records).sort_values(
        "max_attention", ascending=False
    ).reset_index(drop=True)

    # --- Figure ---
    fig_png = args.out_fig_dir / "fig_ccc_outlier_demographics.png"
    fig_pdf = args.out_fig_dir / "fig_ccc_outlier_demographics.pdf"
    _render_figure(
        out_meta=out_meta,
        typ_meta=typ_meta,
        out_pairs=out_pairs,
        typ_pairs=typ_pairs,
        cell_types=cell_types,
        out_path_png=fig_png,
        out_path_pdf=fig_pdf,
    )
    logger.info("Wrote %s", fig_png)
    logger.info("Wrote %s", fig_pdf)

    # --- JSON ---
    args.out_data_dir.mkdir(parents=True, exist_ok=True)
    out_top_pairs_sorted = sorted(out_pairs.items(), key=lambda kv: kv[1], reverse=True)[:20]
    typ_top_pairs_sorted = sorted(typ_pairs.items(), key=lambda kv: kv[1], reverse=True)[:20]
    out_pairs_json = [
        {"source_ct": k[0], "target_ct": k[1], "n_subjects": int(v)}
        for k, v in out_top_pairs_sorted
    ]
    typ_pairs_json = [
        {"source_ct": k[0], "target_ct": k[1], "n_subjects": int(v)}
        for k, v in typ_top_pairs_sorted
    ]
    out_payload = {
        "config": {
            "threshold": args.threshold,
            "top_k_pairs": args.top_k_pairs,
            "n_outliers": int(len(outliers)),
            "n_typical": int(len(typicals)),
            "ccc_summary_json": str(args.ccc_summary_json),
            "metadata_csv": str(args.metadata_csv),
        },
        "enrichment": enrichment,
        "outlier_subjects": [
            {
                **rec,
                "dominant_edges": [
                    {
                        "source_ct": e["source_ct"],
                        "target_ct": e["target_ct"],
                        "edge_type": e["edge_type"],
                        "attention": float(e["attention"]),
                    }
                    for e in dominant.get(rec["subject_id"], [])
                ],
            }
            for rec in outlier_records
        ],
        "outlier_top_ct_pairs": out_pairs_json,
        "typical_top_ct_pairs": typ_pairs_json,
        "url_caveat": {
            "url_appears_in_outlier_edges": bool(url_in_outliers),
            "note": (
                "URL (Upper-rhombic-lip) is sparse (zero_frac ~ 0.79), "
                "so URL involvement in dominant edges is coverage-artifact-affected."
            ),
        },
    }
    json_path = args.out_data_dir / "per_subject_outlier_analysis.json"
    json_path.write_text(json.dumps(out_payload, indent=2))
    logger.info("Wrote %s", json_path)

    # --- Markdown ---
    md = _build_markdown(
        n_outliers=len(outliers),
        n_typical=len(typicals),
        threshold=args.threshold,
        enrichment=enrichment,
        outlier_table=outlier_df,
        out_top_pairs=out_top_pairs_sorted,
        typ_top_pairs=typ_top_pairs_sorted,
        url_caveat=url_in_outliers,
    )
    md_path = args.out_data_dir / "per_subject_outlier_analysis.md"
    md_path.write_text(md)
    logger.info("Wrote %s", md_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
