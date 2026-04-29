"""F3 follow-up: Head-1 Splatter attention vs per-subject Splatter cell count.

Audit Finding 3 noted Head 1 has constitutive ~0.123 attention to Splatter
(3.8x uniform baseline), but per-subject Splatter attention does NOT correlate
with the residual (r = -0.026). Open question: is per-subject Head-1 Splatter
attention modulated by per-subject Splatter cell count?

Approach
--------
For each subject, pull:
  - n_splatter_cells_pfc (or n_splatter_cells_total when PFC count is NaN)
    from splatter_per_subject_features.csv
  - attention[subject_idx, head=1, ct=Splatter] from
    pathology_attention_per_subject.npz (axis 2 ordered per
    pathology_attention_summary.json -> Splatter == index 30)

Compute Spearman + Pearson correlation across all subjects with valid pairs,
plus the same correlation restricted to subjects where Splatter is actually
present in PFC (n_splatter_cells_pfc > 0). Report quartile statistics so we can
see whether the relationship is driven by tails.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# Cell type ordering taken from
# outputs/canonical/interpretability/pathology_attention_summary.json
SPLATTER_CT_INDEX = 30
HEAD1_INDEX = 1


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--features-csv",
        type=Path,
        default=Path("outputs/canonical/interpretability/splatter_per_subject_features.csv"),
    )
    p.add_argument(
        "--attention-npz",
        type=Path,
        default=Path("outputs/canonical/interpretability/pathology_attention_per_subject.npz"),
    )
    p.add_argument(
        "--summary-json",
        type=Path,
        default=Path("outputs/canonical/interpretability/pathology_attention_summary.json"),
    )
    p.add_argument(
        "--output-json",
        type=Path,
        default=Path(
            "outputs/canonical/interpretability/head1_splatter_cell_count_correlation.json"
        ),
    )
    return p.parse_args()


def quartile_summary(x: np.ndarray) -> dict:
    return {
        "min": float(np.min(x)),
        "q1": float(np.percentile(x, 25)),
        "median": float(np.median(x)),
        "q3": float(np.percentile(x, 75)),
        "max": float(np.max(x)),
        "mean": float(np.mean(x)),
        "std": float(np.std(x, ddof=1)) if len(x) > 1 else 0.0,
    }


def correlations(x: np.ndarray, y: np.ndarray) -> dict:
    spearman = stats.spearmanr(x, y)
    pearson = stats.pearsonr(x, y)
    return {
        "n": int(len(x)),
        "spearman_rho": float(spearman.statistic),
        "spearman_p": float(spearman.pvalue),
        "pearson_r": float(pearson.statistic),
        "pearson_p": float(pearson.pvalue),
    }


def main() -> None:
    args = parse_args()

    # --- Verify cell type axis alignment ---
    with args.summary_json.open() as f:
        summary = json.load(f)
    ct_names = summary["cell_type_names_used"]
    if ct_names[SPLATTER_CT_INDEX] != "Splatter":
        raise RuntimeError(
            f"CT index {SPLATTER_CT_INDEX} is {ct_names[SPLATTER_CT_INDEX]!r}, expected 'Splatter'"
        )

    # --- Load features ---
    features = pd.read_csv(args.features_csv)
    # n_splatter_cells_pfc is NaN when subject_id is sampled from non-PFC region;
    # in that case fall back to the total count (PFC is the modeled region).
    cells_pfc = features["n_splatter_cells_pfc"].astype(float)
    cells_total = features["n_splatter_cells_total"].astype(float)
    n_cells = cells_pfc.where(~cells_pfc.isna(), cells_total).to_numpy()

    # --- Load attention tensor ---
    npz = np.load(args.attention_npz, allow_pickle=True)
    subject_ids = np.array([str(s) for s in npz["subject_ids"]])
    attention = npz["attention"]  # (n_subjects, n_heads, n_cell_types)
    if attention.shape[1] <= HEAD1_INDEX or attention.shape[2] <= SPLATTER_CT_INDEX:
        raise RuntimeError(f"Unexpected attention shape: {attention.shape}")
    head1_splatter_attn = attention[:, HEAD1_INDEX, SPLATTER_CT_INDEX].astype(float)

    # --- Align by subject id ---
    feat_subject_ids = features["subject"].astype(str).to_numpy()
    attn_index = {sid: i for i, sid in enumerate(subject_ids)}
    aligned_n_cells = []
    aligned_attn = []
    aligned_residual = []
    aligned_subjects = []
    aligned_present_pfc = []
    for sid, n, resid, present in zip(
        feat_subject_ids,
        n_cells,
        features["residual"].to_numpy(),
        features["splatter_present_in_pfc"].astype(bool).to_numpy(),
    ):
        if sid not in attn_index:
            continue
        idx = attn_index[sid]
        aligned_subjects.append(sid)
        aligned_n_cells.append(float(n))
        aligned_attn.append(float(head1_splatter_attn[idx]))
        aligned_residual.append(float(resid))
        aligned_present_pfc.append(bool(present))

    aligned_n_cells = np.asarray(aligned_n_cells, dtype=float)
    aligned_attn = np.asarray(aligned_attn, dtype=float)
    aligned_residual = np.asarray(aligned_residual, dtype=float)
    aligned_present_pfc = np.asarray(aligned_present_pfc, dtype=bool)

    # --- All subjects ---
    all_corr = correlations(aligned_n_cells, aligned_attn)
    # --- Subjects with Splatter present in PFC ---
    mask_pfc = aligned_present_pfc & (aligned_n_cells > 0)
    pfc_corr = correlations(aligned_n_cells[mask_pfc], aligned_attn[mask_pfc])
    # --- log-transformed (heavy-tailed cell count distribution) ---
    log_corr = correlations(np.log1p(aligned_n_cells), aligned_attn)

    # --- Quartile summaries ---
    cell_quartiles = quartile_summary(aligned_n_cells)
    attn_quartiles = quartile_summary(aligned_attn)

    # --- Per-quartile attention means (binned by cell count) ---
    quartile_bins = np.percentile(aligned_n_cells, [25, 50, 75])
    bin_assignment = np.digitize(aligned_n_cells, quartile_bins)
    by_quartile = []
    for q in range(4):
        mask = bin_assignment == q
        if mask.sum() == 0:
            by_quartile.append({"quartile": q + 1, "n": 0})
            continue
        by_quartile.append(
            {
                "quartile": q + 1,
                "n": int(mask.sum()),
                "n_cells_range": [
                    float(aligned_n_cells[mask].min()),
                    float(aligned_n_cells[mask].max()),
                ],
                "head1_splatter_attn_mean": float(aligned_attn[mask].mean()),
                "head1_splatter_attn_std": float(aligned_attn[mask].std(ddof=1))
                if mask.sum() > 1
                else 0.0,
            }
        )

    # --- Verdict logic ---
    p_threshold = 0.05
    is_significant_either = (
        all_corr["spearman_p"] < p_threshold
        or all_corr["pearson_p"] < p_threshold
        or log_corr["spearman_p"] < p_threshold
        or log_corr["pearson_p"] < p_threshold
    )
    if is_significant_either:
        verdict = (
            "Head-1 Splatter attention IS subject-modulated by Splatter cell count "
            f"(min p across tests = {min(all_corr['spearman_p'], all_corr['pearson_p'], log_corr['spearman_p'], log_corr['pearson_p']):.4g})."
        )
    else:
        verdict = (
            "Head-1 Splatter attention is truly CONSTITUTIVE (not modulated by "
            "per-subject Splatter cell count). All correlations are null at p>=0.05."
        )

    payload = {
        "task": "F3 follow-up: Head-1 attention vs Splatter cell count",
        "splatter_ct_index": SPLATTER_CT_INDEX,
        "head_index": HEAD1_INDEX,
        "all_subjects": all_corr,
        "subjects_with_splatter_in_pfc": pfc_corr,
        "log1p_cell_count": log_corr,
        "cell_count_quartile_summary": cell_quartiles,
        "head1_splatter_attention_summary": attn_quartiles,
        "by_cell_count_quartile": by_quartile,
        "verdict": verdict,
        "per_subject_pairs": [
            {
                "subject": sid,
                "n_splatter_cells": float(n),
                "head1_splatter_attention": float(a),
                "residual": float(r),
                "splatter_present_in_pfc": bool(p),
            }
            for sid, n, a, r, p in zip(
                aligned_subjects,
                aligned_n_cells,
                aligned_attn,
                aligned_residual,
                aligned_present_pfc,
            )
        ],
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w") as f:
        json.dump(payload, f, indent=2)

    print(f"n_subjects (all): {all_corr['n']}")
    print(f"n_subjects (Splatter in PFC): {pfc_corr['n']}")
    print(
        f"All subjects   : Spearman rho = {all_corr['spearman_rho']:+.4f} (p = {all_corr['spearman_p']:.4g}); "
        f"Pearson r = {all_corr['pearson_r']:+.4f} (p = {all_corr['pearson_p']:.4g})"
    )
    print(
        f"PFC-present    : Spearman rho = {pfc_corr['spearman_rho']:+.4f} (p = {pfc_corr['spearman_p']:.4g}); "
        f"Pearson r = {pfc_corr['pearson_r']:+.4f} (p = {pfc_corr['pearson_p']:.4g})"
    )
    print(
        f"log1p(cells)   : Spearman rho = {log_corr['spearman_rho']:+.4f} (p = {log_corr['spearman_p']:.4g}); "
        f"Pearson r = {log_corr['pearson_r']:+.4f} (p = {log_corr['pearson_p']:.4g})"
    )
    print(f"VERDICT: {verdict}")
    print(f"Wrote: {args.output_json}")


if __name__ == "__main__":
    main()
