"""Resilience residual phenotyping from canonical ResDec-H3 predictions.

Loads ``val_predictions_best.npz`` across all 5 folds (canonical n_stages=1 +
TabM run), joins to ROSMAP metadata, and computes per-subject **resilience
residual**:

    residual_i = target_i − prediction_i

Sign convention:
    residual > 0  → subject has BETTER cognition than the model predicted
                    given their gene-expression / CCC / pathology profile
                    → "more cognitively resilient than expected"
    residual < 0  → subject has WORSE cognition than predicted → "more
                    vulnerable than expected"

Outputs (default ``outputs/redesign/interpretability/``):
  - residual_per_subject.csv  — full per-subject table (metadata + residual)
  - residual_summary.json     — distribution stats + APOE / sex / age / pathology breakdowns
  - top_resilient.csv         — top-20 largest positive residual (the resilient phenotype)
  - top_vulnerable.csv        — top-20 largest negative residual

Usage
-----
    PYTHONPATH=<worktree-root> \\
    uv run python scripts/redesign/interpretability/resilience_residual_phenotype.py \\
        --pred-root outputs/redesign/p5_phase3_1stage_with_tabm \\
        --out-dir outputs/redesign/interpretability

Arguments
---------
    --pred-root <path>     Directory containing ``fold{0..4}/val_predictions_best.npz``.
                           Default points at the canonical seed-42 run.
    --metadata-csv <path>  ROSMAP metadata CSV (default ``data/metadata_ROSMAP/metadata.csv``).
                           Join key: ``ROSMAP_IndividualID``.
    --out-dir <path>       Output directory (created if missing).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Make the script standalone-runnable: ensure the worktree root is on sys.path.
# Anchored at parents[3] (i.e. scripts/redesign/interpretability/<this_file> → worktree_root/).
_WORKTREE_ROOT = Path(__file__).resolve().parents[3]
if not (_WORKTREE_ROOT / "src").is_dir():
    raise RuntimeError(
        f"sys.path bootstrap failed: {_WORKTREE_ROOT}/src not found; "
        "set PYTHONPATH=<worktree-root>."
    )
sys.path.insert(0, str(_WORKTREE_ROOT))


def load_all_folds(pred_root: Path) -> pd.DataFrame:
    """Concatenate val_predictions_best.npz across all 5 outer folds."""
    rows: list[dict] = []
    for f in range(5):
        p = pred_root / f"fold{f}/val_predictions_best.npz"
        if not p.exists():
            raise FileNotFoundError(
                f"Missing per-fold predictions: {p}. Run the canonical 5-fold + "
                f"reinfer pipeline first (scripts/redesign/run_phase2_5fold_parallel.sh)."
            )
        d = np.load(p, allow_pickle=True)
        sids = d["subject_ids"].astype(str)
        preds = d["predictions"].astype(np.float32)
        targets = d["targets"].astype(np.float32)
        for sid, pred, t in zip(sids, preds, targets):
            rows.append({
                "ROSMAP_IndividualID": str(sid),
                "fold": int(f),
                "target": float(t),
                "prediction": float(pred),
                "residual": float(t - pred),
            })
    return pd.DataFrame(rows)


def _quartile(df: pd.DataFrame, col: str, q: int = 4) -> pd.Series:
    """Quartile-rank a column (NaN-safe via rank(method='first'))."""
    return pd.qcut(df[col].rank(method="first"), q=q,
                   labels=[f"Q{i + 1}" for i in range(q)])


def summarize(df: pd.DataFrame) -> dict:
    """Distribution + subgroup breakdowns of the residual."""
    summary: dict = {
        "n_subjects": int(len(df)),
        "residual_mean": float(df["residual"].mean()),
        "residual_std": float(df["residual"].std()),
        "residual_quantiles": {
            "p10": float(df["residual"].quantile(0.10)),
            "Q1": float(df["residual"].quantile(0.25)),
            "median": float(df["residual"].median()),
            "Q3": float(df["residual"].quantile(0.75)),
            "p90": float(df["residual"].quantile(0.90)),
        },
    }

    # APOE ε4 carrier-count breakdown.
    if "apoe_genotype" in df.columns:
        df = df.copy()
        df["apoe_e4_count"] = (
            df["apoe_genotype"].astype(str).apply(lambda x: x.count("4"))
        )
        ap = df.groupby("apoe_e4_count")["residual"].agg(["mean", "std", "count"])
        summary["by_apoe_e4_count"] = {
            int(k): {kk: float(vv) for kk, vv in v.items()} for k, v in ap.to_dict("index").items()
        }

    # Sex breakdown (msex: 0 = female, 1 = male in ROSMAP).
    if "msex" in df.columns:
        sex = df.groupby("msex")["residual"].agg(["mean", "std", "count"])
        summary["by_msex"] = {
            int(k): {kk: float(vv) for kk, vv in v.items()} for k, v in sex.to_dict("index").items()
        }

    # Age-at-death quartiles.
    if "age_death" in df.columns and df["age_death"].notna().any():
        df_age = df.dropna(subset=["age_death"]).copy()
        df_age["age_quartile"] = _quartile(df_age, "age_death")
        ag = df_age.groupby("age_quartile", observed=True)["residual"].agg(["mean", "std", "count"])
        summary["by_age_quartile"] = {
            str(k): {kk: float(vv) for kk, vv in v.items()} for k, v in ag.to_dict("index").items()
        }

    # Pathology correlations: how does residual relate to AD pathology?
    # Negative correlation → subjects with high pathology tend to be MORE
    # vulnerable than predicted (model over-credits their cognition);
    # positive → high pathology + high residual = unexpectedly resilient.
    pathology_cols = ("amyloid", "tangles", "gpath", "braaksc", "ceradsc",
                      "niareagansc", "plaq_n_mf")
    summary["residual_pathology_correlations"] = {}
    for col in pathology_cols:
        if col in df.columns and df[col].notna().sum() > 30:
            sub = df.dropna(subset=["residual", col])
            corr_p = float(sub["residual"].corr(sub[col]))
            corr_s = float(sub["residual"].corr(sub[col], method="spearman"))
            summary["residual_pathology_correlations"][col] = {
                "pearson_r": corr_p, "spearman_rho": corr_s, "n": int(len(sub)),
            }

    # "Resilient phenotype" tag: high pathology AND high cognition (positive residual
    # AND above-median pathology). This is the canonical AD-resilience definition.
    if "amyloid" in df.columns:
        median_amyloid = df["amyloid"].median()
        median_resid = df["residual"].median()
        df_phen = df.dropna(subset=["amyloid", "residual"]).copy()
        df_phen["high_pathology"] = df_phen["amyloid"] > median_amyloid
        df_phen["positive_residual"] = df_phen["residual"] > median_resid
        cross = pd.crosstab(df_phen["high_pathology"], df_phen["positive_residual"])
        summary["resilience_quadrants"] = {
            "high_pathology_high_residual_RESILIENT": int(cross.loc[True, True]) if (True in cross.index and True in cross.columns) else 0,
            "high_pathology_low_residual_VULNERABLE": int(cross.loc[True, False]) if (True in cross.index and False in cross.columns) else 0,
            "low_pathology_high_residual": int(cross.loc[False, True]) if (False in cross.index and True in cross.columns) else 0,
            "low_pathology_low_residual": int(cross.loc[False, False]) if (False in cross.index and False in cross.columns) else 0,
            "median_amyloid_threshold": float(median_amyloid),
            "median_residual_threshold": float(median_resid),
        }

    return summary


def main(args: argparse.Namespace) -> int:
    pred_root = Path(args.pred_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_pred = load_all_folds(pred_root)
    print(f"Loaded {len(df_pred)} subjects across {df_pred['fold'].nunique()} folds")

    meta_csv = Path(args.metadata_csv)
    if not meta_csv.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {meta_csv}")
    meta = pd.read_csv(meta_csv)
    if "ROSMAP_IndividualID" not in meta.columns:
        raise KeyError("metadata.csv missing ROSMAP_IndividualID column")

    df = df_pred.merge(meta, on="ROSMAP_IndividualID", how="left")
    n_with_meta = df["apoe_genotype"].notna().sum() if "apoe_genotype" in df else 0
    print(f"After metadata join: {len(df)} subjects, {n_with_meta} with APOE genotype")

    # Per-subject CSV.
    out_csv = out_dir / "residual_per_subject.csv"
    df.to_csv(out_csv, index=False)

    # Summary JSON.
    summary = summarize(df)
    summary_path = out_dir / "residual_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=float))

    # Top-resilient / top-vulnerable tables.
    cols_keep = ["ROSMAP_IndividualID", "fold", "target", "prediction", "residual",
                 "apoe_genotype", "msex", "age_death",
                 "amyloid", "tangles", "braaksc", "ceradsc", "gpath"]
    cols_keep = [c for c in cols_keep if c in df.columns]
    df_sorted = df.sort_values("residual")
    df_sorted.tail(20)[cols_keep].iloc[::-1].to_csv(out_dir / "top_resilient.csv", index=False)
    df_sorted.head(20)[cols_keep].to_csv(out_dir / "top_vulnerable.csv", index=False)

    print(f"\nWrote {out_csv}")
    print(f"      {summary_path}")
    print(f"      {out_dir / 'top_resilient.csv'}")
    print(f"      {out_dir / 'top_vulnerable.csv'}")
    print()
    print("=== Summary (truncated; full in residual_summary.json) ===")
    print(f"  n_subjects: {summary['n_subjects']}")
    print(f"  residual: mean={summary['residual_mean']:+.4f} std={summary['residual_std']:.4f}")
    if "residual_pathology_correlations" in summary:
        print("  Residual ↔ pathology Pearson r:")
        for k, v in summary["residual_pathology_correlations"].items():
            print(f"    {k:<14s}: r = {v['pearson_r']:+.4f} (n={v['n']})")
    if "resilience_quadrants" in summary:
        rq = summary["resilience_quadrants"]
        print(f"  Resilient quadrant (high pathology + positive residual): "
              f"n = {rq['high_pathology_high_residual_RESILIENT']}")
        print(f"  Vulnerable quadrant (high pathology + negative residual): "
              f"n = {rq['high_pathology_low_residual_VULNERABLE']}")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Residual phenotyping for ResDec-H3.")
    p.add_argument("--pred-root", default="outputs/redesign/p5_phase3_1stage_with_tabm",
                   help="Directory containing fold{0..4}/val_predictions_best.npz")
    p.add_argument("--metadata-csv", default="data/metadata_ROSMAP/metadata.csv")
    p.add_argument("--out-dir", default="outputs/redesign/interpretability")
    sys.exit(main(p.parse_args()))
