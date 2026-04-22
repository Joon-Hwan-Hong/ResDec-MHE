"""Per-head analysis of PathologyAttention + Splatter deep-dive.

Loads ``pathology_attention_per_subject.npz`` (produced by
``extract_pathology_attention.py``) and answers:

  1. **Head specialization**: what does each of the 4 attention heads capture?
     - Top-3 cell types per head (mean attention).
     - Per-head Shannon entropy (lower = more specialized; higher = more uniform).
     - Per-head effective n  ( 1 / Σ_c p_c² ; range [1, n_cell_types] ).

  2. **Inter-head redundancy**: are heads complementary or duplicating?
     - Pairwise cosine similarity of mean-attention vectors across heads.
     - Cosine on the full per-subject head-vectors → average inter-head agreement.

  3. **Subject-level head heterogeneity**: do different subjects use different
     head-mixes? If yes → heads encode something subject-specific (interesting).
     If no → heads are a bias-only effect.
     - Per-subject head-attention "fingerprint" = mean of attention over cell
       types per head [N, n_heads]; std across subjects per head measures
       variability.
     - Correlate per-subject head-fingerprint with metadata (APOE, sex, age,
       pathology, residual from prior phenotype script).

  4. **Splatter deep-dive** (motivated by Siletti et al. 2023 — Splatter in PFC
     samples is dominated by long-range SST+CHODL+ GABAergic projection
     interneurons, NOT noise):
     - Per-subject Splatter attention vs cognition / pathology / resilience
       residual.
     - Co-attention with other GABAergic interneurons (LAMP5-LHX6 + Chandelier,
       MGE, CGE) — if positively correlated, the model is consistently weighting
       inhibitory neurons together.

Outputs (default ``outputs/redesign/interpretability/``):
  - head_analysis_summary.json        — head specialization + redundancy metrics
  - splatter_deepdive_summary.json    — Splatter biology breakdown
  - per_subject_head_fingerprints.csv — N × (n_heads + metadata cols)

Usage
-----
    PYTHONPATH=<worktree-root> \\
    uv run python scripts/redesign/interpretability/analyze_pathology_attention_heads.py \\
        --attn-npz outputs/redesign/interpretability/pathology_attention_per_subject.npz \\
        --residual-csv outputs/redesign/interpretability/residual_per_subject.csv \\
        --out-dir outputs/redesign/interpretability
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_WORKTREE_ROOT = Path(__file__).resolve().parents[2]
if not (_WORKTREE_ROOT / "src").is_dir():
    raise RuntimeError(
        f"sys.path bootstrap failed: {_WORKTREE_ROOT}/src not found; "
        "set PYTHONPATH=<worktree-root>."
    )
sys.path.insert(0, str(_WORKTREE_ROOT))

from src.data.constants import CELL_TYPE_ORDER

# Cell types we care about for the GABAergic interneuron co-attention check.
# Splatter in PFC = long-range SST+CHODL+ projection GABAergic neurons (Siletti
# et al. 2023). LAMP5-LHX6 + Chandelier are upper-layer cortical GABAergic
# interneurons. MGE and CGE are the developmental origins of most cortical
# inhibitory neurons.
GABAERGIC_CELL_TYPES = (
    "Splatter",
    "LAMP5-LHX6 and Chandelier",
    "MGE interneuron",
    "CGE interneuron",
)


def shannon_entropy(p: np.ndarray) -> float:
    """Shannon entropy in nats. Robust to p containing zeros."""
    p = np.asarray(p, dtype=np.float64)
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


def effective_n(p: np.ndarray) -> float:
    """Effective number of contributors: 1 / Σ p²  ∈ [1, len(p)]."""
    p = np.asarray(p, dtype=np.float64)
    return float(1.0 / np.sum(p * p))


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def head_specialization(attn_mean_per_head: np.ndarray, ct_names: list[str]) -> list[dict]:
    """Top cell types + entropy + effective_n per head.

    attn_mean_per_head: [n_heads, n_cell_types] — mean attention across subjects.
    """
    out: list[dict] = []
    n_heads, n_ct = attn_mean_per_head.shape
    for h in range(n_heads):
        p = attn_mean_per_head[h]
        # Normalise — mean attention should sum close to 1 already, but be safe.
        p_norm = p / max(p.sum(), 1e-12)
        order = np.argsort(-p_norm)
        out.append({
            "head": h,
            "shannon_entropy_nats": shannon_entropy(p_norm),
            "effective_n_cell_types": effective_n(p_norm),
            "top_3_cell_types": [
                {"cell_type": ct_names[c], "mean_attention": float(p_norm[c])}
                for c in order[:3]
            ],
        })
    return out


def inter_head_redundancy(attn_mean_per_head: np.ndarray) -> dict:
    """Pairwise cosine of mean-attention vectors across heads.

    Cosine ≈ 1 → heads are redundant; cosine ≈ 0 → orthogonal/complementary.
    """
    n_heads = attn_mean_per_head.shape[0]
    pairs = {}
    for i in range(n_heads):
        for j in range(i + 1, n_heads):
            pairs[f"h{i}_vs_h{j}"] = cosine_similarity(
                attn_mean_per_head[i], attn_mean_per_head[j]
            )
    sims = list(pairs.values())
    return {
        "pairwise_cosine": pairs,
        "mean_pairwise_cosine": float(np.mean(sims)) if sims else 0.0,
        "max_pairwise_cosine": float(np.max(sims)) if sims else 0.0,
        "min_pairwise_cosine": float(np.min(sims)) if sims else 0.0,
    }


def subject_level_head_fingerprints(attn: np.ndarray) -> np.ndarray:
    """Per-subject, per-head attention magnitude (sum over cell types).

    Returns [N, n_heads] — each entry is the head's total attention budget for
    that subject. Useful as a head-usage 'fingerprint' to correlate with metadata.
    """
    return attn.sum(axis=2)  # [N, n_heads]


def splatter_deepdive(attn: np.ndarray, ct_names: list[str], sids: np.ndarray,
                      residual_df: pd.DataFrame) -> dict:
    """Splatter attention vs cognition / pathology / resilience residual,
    plus co-attention with other GABAergic populations."""
    n_heads = attn.shape[1]
    if "Splatter" not in ct_names:
        return {"error": "Splatter cell type not in cell_type_names_used"}
    splatter_idx = ct_names.index("Splatter")
    splatter_attn = attn[:, :, splatter_idx]  # [N, n_heads]
    splatter_total = splatter_attn.sum(axis=1)  # [N] aggregated across heads

    # Build per-subject DataFrame: subject_id, splatter_total, splatter_per_head_*
    df_attn = pd.DataFrame({
        "ROSMAP_IndividualID": sids.astype(str),
        "splatter_attn_total": splatter_total,
    })
    for h in range(n_heads):
        df_attn[f"splatter_attn_h{h}"] = splatter_attn[:, h]

    # Co-attention with other GABAergic interneurons: per-subject summed
    # attention across heads for each cell type.
    coatt = {}
    for ct in GABAERGIC_CELL_TYPES:
        if ct not in ct_names:
            continue
        idx = ct_names.index(ct)
        coatt[ct] = attn[:, :, idx].sum(axis=1)  # [N]
        df_attn[f"attn_total_{ct.replace(' ', '_')}"] = coatt[ct]

    # Pairwise correlations among GABAergic populations.
    coatt_corrs: dict = {}
    keys = list(coatt.keys())
    for i, k1 in enumerate(keys):
        for k2 in keys[i + 1:]:
            r = float(np.corrcoef(coatt[k1], coatt[k2])[0, 1])
            coatt_corrs[f"{k1}__vs__{k2}"] = r

    # Join to residual / metadata for biology correlations.
    out: dict = {
        "splatter_attn_total_stats": {
            "mean": float(splatter_total.mean()),
            "std": float(splatter_total.std()),
            "min": float(splatter_total.min()),
            "max": float(splatter_total.max()),
        },
        "gabaergic_interneuron_co_attention_pearson_r": coatt_corrs,
    }

    if residual_df is not None:
        merged = residual_df.merge(df_attn, on="ROSMAP_IndividualID", how="inner")
        for col, label in [
            ("residual", "splatter_attn_vs_resilience_residual"),
            ("cogn_global", "splatter_attn_vs_cognition"),
            ("amyloid", "splatter_attn_vs_amyloid"),
            ("tangles", "splatter_attn_vs_tangles"),
            ("braaksc", "splatter_attn_vs_braaksc"),
        ]:
            if col in merged.columns and merged[col].notna().sum() > 30:
                sub = merged.dropna(subset=["splatter_attn_total", col])
                pearson = float(sub["splatter_attn_total"].corr(sub[col]))
                spearman = float(sub["splatter_attn_total"].corr(sub[col], method="spearman"))
                out[label] = {
                    "pearson_r": pearson,
                    "spearman_rho": spearman,
                    "n": int(len(sub)),
                }

    return out


def head_metadata_correlations(fingerprint: np.ndarray, sids: np.ndarray,
                               residual_df: pd.DataFrame) -> dict:
    """For each head, correlate per-subject head-attention-total against
    APOE / age / pathology / residual."""
    n_heads = fingerprint.shape[1]
    df = pd.DataFrame({"ROSMAP_IndividualID": sids.astype(str)})
    for h in range(n_heads):
        df[f"head_{h}_total"] = fingerprint[:, h]

    if residual_df is None:
        return {"error": "residual_df not available; skipping head ↔ metadata correlations"}

    merged = residual_df.merge(df, on="ROSMAP_IndividualID", how="inner")
    if "apoe_genotype" in merged.columns:
        merged = merged.copy()
        merged["apoe_e4_count"] = merged["apoe_genotype"].astype(str).apply(
            lambda x: x.count("4")
        )

    out: dict = {}
    metadata_cols = ("residual", "cogn_global", "apoe_e4_count", "msex",
                     "age_death", "amyloid", "tangles", "braaksc")
    for h in range(n_heads):
        head_col = f"head_{h}_total"
        head_corrs: dict = {}
        for col in metadata_cols:
            if col in merged.columns and merged[col].notna().sum() > 30:
                sub = merged.dropna(subset=[head_col, col])
                head_corrs[col] = {
                    "pearson_r": float(sub[head_col].corr(sub[col])),
                    "spearman_rho": float(sub[head_col].corr(sub[col], method="spearman")),
                    "n": int(len(sub)),
                }
        out[f"head_{h}"] = head_corrs
    return out


def main(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    attn_npz = Path(args.attn_npz)
    if not attn_npz.exists():
        raise FileNotFoundError(f"Attention .npz not found: {attn_npz}. "
                                "Run extract_pathology_attention.py first.")
    npz = np.load(attn_npz, allow_pickle=True)
    attn = npz["attention"]  # [N, H, C]
    sids = npz["subject_ids"]
    folds = npz["fold"]

    n_subj, n_heads, n_ct = attn.shape
    print(f"Loaded attention {attn.shape} from {attn_npz}")

    # cell_type_names: prefer the names saved in the matching summary; fall back
    # to constants. Match length to actual n_ct (truncate if constants longer).
    summary_path = attn_npz.parent / "pathology_attention_summary.json"
    if summary_path.exists():
        names = json.loads(summary_path.read_text())["cell_type_names_used"]
    else:
        names = list(CELL_TYPE_ORDER)
    ct_names = names[:n_ct]
    if len(ct_names) < n_ct:
        ct_names = ct_names + [f"ct_{c}" for c in range(len(ct_names), n_ct)]

    # Optional residual CSV from the prior phenotype script — joined for biology.
    residual_df: pd.DataFrame | None = None
    rcsv = Path(args.residual_csv) if args.residual_csv else None
    if rcsv is not None and rcsv.exists():
        residual_df = pd.read_csv(rcsv)
        print(f"Joined residual table: {len(residual_df)} subjects from {rcsv}")
    else:
        print(f"No residual CSV provided or found ({rcsv}); skipping head↔metadata correlations.")

    attn_mean_per_head = attn.mean(axis=0)  # [H, C]

    head_summary = head_specialization(attn_mean_per_head, ct_names)
    redundancy = inter_head_redundancy(attn_mean_per_head)
    fingerprint = subject_level_head_fingerprints(attn)
    fp_stats = {
        f"head_{h}_subject_total_mean": float(fingerprint[:, h].mean())
        for h in range(n_heads)
    }
    fp_stats.update({
        f"head_{h}_subject_total_std": float(fingerprint[:, h].std())
        for h in range(n_heads)
    })

    head_meta_corr = head_metadata_correlations(fingerprint, sids, residual_df)
    splatter = splatter_deepdive(attn, ct_names, sids, residual_df)

    head_summary_json = {
        "n_subjects": int(n_subj),
        "n_heads": int(n_heads),
        "n_cell_types": int(n_ct),
        "uniform_baseline_per_cell_type": float(1.0 / n_ct),
        "max_entropy_nats": float(np.log(n_ct)),
        "head_specialization": head_summary,
        "inter_head_redundancy": redundancy,
        "subject_head_total_stats": fp_stats,
        "head_vs_metadata_correlations": head_meta_corr,
    }
    (out_dir / "head_analysis_summary.json").write_text(
        json.dumps(head_summary_json, indent=2, default=float)
    )
    (out_dir / "splatter_deepdive_summary.json").write_text(
        json.dumps(splatter, indent=2, default=float)
    )

    # Per-subject head fingerprint CSV.
    fp_df = pd.DataFrame({"ROSMAP_IndividualID": sids.astype(str), "fold": folds})
    for h in range(n_heads):
        fp_df[f"head_{h}_total"] = fingerprint[:, h]
    fp_df.to_csv(out_dir / "per_subject_head_fingerprints.csv", index=False)

    print(f"\nWrote {out_dir / 'head_analysis_summary.json'}")
    print(f"      {out_dir / 'splatter_deepdive_summary.json'}")
    print(f"      {out_dir / 'per_subject_head_fingerprints.csv'}")

    print()
    print("=== Head specialization ===")
    print(f"max entropy (uniform 31 CTs) = {np.log(n_ct):.3f} nats")
    for h in head_summary:
        top3 = ", ".join(f"{e['cell_type']}={e['mean_attention']:.3f}"
                         for e in h["top_3_cell_types"])
        print(f"  head {h['head']}: H={h['shannon_entropy_nats']:.3f} nats "
              f"(eff_n={h['effective_n_cell_types']:.2f}); top: {top3}")

    print()
    print("=== Inter-head redundancy (cosine of mean-attention vectors) ===")
    for k, v in redundancy["pairwise_cosine"].items():
        print(f"  {k}: cos = {v:+.4f}")
    print(f"  mean = {redundancy['mean_pairwise_cosine']:+.4f}, "
          f"max = {redundancy['max_pairwise_cosine']:+.4f}, "
          f"min = {redundancy['min_pairwise_cosine']:+.4f}")

    print()
    print("=== Splatter deep-dive ===")
    print(f"  splatter_attn_total: mean={splatter['splatter_attn_total_stats']['mean']:.4f}  "
          f"std={splatter['splatter_attn_total_stats']['std']:.4f}")
    print("  GABAergic interneuron co-attention pearson r:")
    for k, v in splatter["gabaergic_interneuron_co_attention_pearson_r"].items():
        print(f"    {k}: {v:+.4f}")
    if residual_df is not None:
        for k in ("splatter_attn_vs_resilience_residual",
                  "splatter_attn_vs_cognition",
                  "splatter_attn_vs_amyloid",
                  "splatter_attn_vs_tangles",
                  "splatter_attn_vs_braaksc"):
            if k in splatter:
                v = splatter[k]
                print(f"    {k}: r={v['pearson_r']:+.4f}  ρ={v['spearman_rho']:+.4f}  n={v['n']}")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--attn-npz",
                   default="outputs/redesign/interpretability/pathology_attention_per_subject.npz")
    p.add_argument("--residual-csv",
                   default="outputs/redesign/interpretability/residual_per_subject.csv")
    p.add_argument("--out-dir", default="outputs/redesign/interpretability")
    sys.exit(main(p.parse_args()))
