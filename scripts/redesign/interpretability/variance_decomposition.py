"""Variance decomposition of the ResDec-H3 composite prediction.

Loads per-fold val predictions (composite ``y_hat = y_tabpfn + f_1``) and the
frozen outer-fold TabPFN-2.6 predictions, recovers the neural-head residual
``f_1 = y_hat - y_tabpfn`` per subject, joins with ROSMAP metadata, and writes
a JSON report with the variance budget decomposed as

    Var(y) = Var(y_tabpfn) + Var(f_1) + 2 Cov(y_tabpfn, f_1) + Var(resid)

globally and stratified by APOE-ε4 count, sex, and age-at-death quartile.

Usage
-----
    PYTHONPATH=<worktree-root> \\
    uv run python scripts/redesign/interpretability/variance_decomposition.py \\
        --pred-root outputs/redesign/p5_canonical_seed42 \\
        --tabpfn-dir data/redesign \\
        --metadata-csv data/metadata_ROSMAP/metadata.csv \\
        --out-dir outputs/redesign/interpretability

Outputs ``<out-dir>/variance_decomposition.json``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Make the script standalone-runnable: ensure the worktree root is on sys.path.
_WORKTREE_ROOT = Path(__file__).resolve().parents[3]
if not (_WORKTREE_ROOT / "src").is_dir():
    raise RuntimeError(
        f"sys.path bootstrap failed: {_WORKTREE_ROOT}/src not found; "
        "set PYTHONPATH=<worktree-root>."
    )
sys.path.insert(0, str(_WORKTREE_ROOT))

from src.analysis.resdec_variance_decomposition import decompose_variance  # noqa: E402


N_FOLDS = 5


def _load_fold_predictions(
    pred_root: Path, tabpfn_dir: Path, fold: int,
) -> pd.DataFrame:
    """Load predictions + TabPFN for a single fold and align by subject_id.

    Returns a long DataFrame with columns
    ``ROSMAP_IndividualID, fold, y_true, y_composite, y_tabpfn, f1_residual``.
    """
    pred_path = pred_root / f"fold{fold}/val_predictions_best.npz"
    tabpfn_path = tabpfn_dir / f"tabpfn_outer_fold{fold}.npz"

    if not pred_path.exists():
        raise FileNotFoundError(f"Missing per-fold predictions: {pred_path}")
    if not tabpfn_path.exists():
        raise FileNotFoundError(f"Missing outer TabPFN file: {tabpfn_path}")

    pred = np.load(pred_path, allow_pickle=True)
    tab = np.load(tabpfn_path, allow_pickle=True)

    pred_df = pd.DataFrame({
        "ROSMAP_IndividualID": pred["subject_ids"].astype(str),
        "y_true": pred["targets"].astype(np.float64),
        "y_composite": pred["predictions"].astype(np.float64),
    })
    tab_df = pd.DataFrame({
        "ROSMAP_IndividualID": tab["val_subject_ids"].astype(str),
        "y_true_tabpfn": tab["y_true"].astype(np.float64),
        "y_tabpfn": tab["y_tabpfn"].astype(np.float64),
    })
    merged = pred_df.merge(tab_df, on="ROSMAP_IndividualID", how="inner")

    missing_in_tabpfn = set(pred_df["ROSMAP_IndividualID"]) - set(tab_df["ROSMAP_IndividualID"])
    if missing_in_tabpfn:
        raise RuntimeError(
            f"Fold {fold}: {len(missing_in_tabpfn)} predicted subjects absent from "
            f"TabPFN outer-fold file ({tabpfn_path.name}). "
            f"First few: {sorted(missing_in_tabpfn)[:5]}"
        )

    # Sanity: y_true in predictions file should match y_true in TabPFN file.
    delta = np.max(np.abs(merged["y_true"].values - merged["y_true_tabpfn"].values))
    if delta > 1e-5:
        raise RuntimeError(
            f"Fold {fold}: y_true mismatch between val_predictions_best.npz and "
            f"tabpfn_outer_fold{fold}.npz (max |Δ| = {delta:.3e}). Refusing to proceed."
        )

    merged["f1_residual"] = merged["y_composite"].values - merged["y_tabpfn"].values
    merged["fold"] = fold
    return merged[
        ["ROSMAP_IndividualID", "fold", "y_true", "y_composite", "y_tabpfn", "f1_residual"]
    ]


def _load_all_folds(pred_root: Path, tabpfn_dir: Path) -> pd.DataFrame:
    """Concatenate all 5 folds' val predictions into a single long DataFrame."""
    return pd.concat(
        [_load_fold_predictions(pred_root, tabpfn_dir, f) for f in range(N_FOLDS)],
        ignore_index=True,
    )


def _apoe_e4_count(genotype: object) -> object:
    """Return the number of ε4 alleles in an APOE genotype string (0/1/2) or None."""
    if genotype is None:
        return None
    try:
        g_float = float(genotype)
        if np.isnan(g_float):
            return None
        g_str = str(int(g_float))
    except (TypeError, ValueError):
        g_str = str(genotype)
    # APOE genotypes are encoded as two-digit concatenations of allele numbers
    # (22, 23, 24, 33, 34, 44) — the number of "4"s is the ε4 count.
    return g_str.count("4")


def _age_quartile_labels(ages: pd.Series) -> pd.Series:
    """Assign Q1..Q4 labels using pandas quantiles on the non-null subset.

    Entries with NaN age receive ``None`` so the downstream subgroup logic drops them.
    """
    labels = pd.Series([None] * len(ages), index=ages.index, dtype=object)
    valid = ages.notna()
    if valid.sum() >= 4:
        # rank(method='first') breaks ties so qcut always gets 4 equal-sized buckets.
        q = pd.qcut(
            ages.loc[valid].rank(method="first"),
            q=4,
            labels=[f"Q{i + 1}" for i in range(4)],
        )
        labels.loc[valid] = q.astype(str).to_numpy()
    return labels


def _build_subgroups(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Build the three stratifications required by the spec."""
    # APOE-ε4 count → "0" / "1" / "2"; None when APOE genotype is missing.
    apoe_str = df["apoe_genotype"].apply(
        lambda g: (lambda c: str(c) if c is not None else None)(_apoe_e4_count(g))
    )

    # Sex: msex ∈ {0, 1}. Keep NaN → None.
    msex_str = df["msex"].apply(
        lambda x: str(int(x)) if pd.notna(x) else None
    )

    age_q = _age_quartile_labels(df["age_death"])

    return {
        "by_apoe_e4_count": apoe_str.to_numpy(dtype=object),
        "by_msex": msex_str.to_numpy(dtype=object),
        "by_age_quartile": age_q.to_numpy(dtype=object),
    }


def _print_summary(decomposition: dict) -> None:
    """Human-readable stdout summary (global fractions + per-subgroup counts)."""
    g = decomposition["global"]
    print("\n=== Variance Decomposition (global) ===")
    print(f"  n                      : {g['n']}")
    print(f"  Var(y)                 : {g['var_y']:.4f}")
    print(f"  Var(y_tabpfn)          : {g['var_tabpfn']:.4f}  "
          f"({g['var_tabpfn'] / g['var_y']:.1%} of Var(y))")
    print(f"  Var(f_1)               : {g['var_f1']:.4f}  "
          f"({g['var_f1'] / g['var_y']:.1%} of Var(y))")
    print(f"  2 Cov(y_tabpfn, f_1)   : {2 * g['cov_tabpfn_f1']:.4f}  "
          f"({2 * g['cov_tabpfn_f1'] / g['var_y']:.1%} of Var(y))")
    print(f"  Var(resid)             : {g['var_resid']:.4f}  "
          f"({g['var_resid'] / g['var_y']:.1%} of Var(y))")
    print(f"  Total explained (1 - Var(resid)/Var(y)) : "
          f"{g['total_explained_fraction']:.4f}")

    for key in ("by_apoe_e4_count", "by_msex", "by_age_quartile"):
        if key not in decomposition:
            continue
        print(f"\n  --- {key} ---")
        for label, stats in sorted(decomposition[key].items(), key=lambda kv: str(kv[0])):
            frac = stats["total_explained_fraction"]
            frac_s = f"{frac:.3f}" if frac == frac else "nan"
            print(f"    {label:<6s}: n={stats['n']:4d}  "
                  f"Var(y)={stats['var_y']:.4f}  "
                  f"Var(resid)={stats['var_resid']:.4f}  "
                  f"total_explained={frac_s}")


def main(args: argparse.Namespace) -> int:
    pred_root = Path(args.pred_root)
    tabpfn_dir = Path(args.tabpfn_dir)
    metadata_csv = Path(args.metadata_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[variance_decomposition] pred-root   = {pred_root}")
    print(f"[variance_decomposition] tabpfn-dir  = {tabpfn_dir}")
    print(f"[variance_decomposition] metadata    = {metadata_csv}")
    print(f"[variance_decomposition] out-dir     = {out_dir}")

    df_pred = _load_all_folds(pred_root, tabpfn_dir)
    print(f"[variance_decomposition] loaded {len(df_pred)} subjects across "
          f"{df_pred['fold'].nunique()} folds")

    if not metadata_csv.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {metadata_csv}")
    meta = pd.read_csv(metadata_csv)
    if "ROSMAP_IndividualID" not in meta.columns:
        raise KeyError("metadata.csv missing ROSMAP_IndividualID column")

    df = df_pred.merge(
        meta[["ROSMAP_IndividualID", "apoe_genotype", "msex", "age_death"]],
        on="ROSMAP_IndividualID", how="left",
    )
    n_apoe = df["apoe_genotype"].notna().sum()
    n_sex = df["msex"].notna().sum()
    n_age = df["age_death"].notna().sum()
    print(f"[variance_decomposition] metadata join: "
          f"APOE available for {n_apoe}/{len(df)}, "
          f"msex for {n_sex}/{len(df)}, "
          f"age_death for {n_age}/{len(df)}")

    subgroups = _build_subgroups(df)

    decomposition = decompose_variance(
        df["y_true"].to_numpy(),
        df["y_tabpfn"].to_numpy(),
        df["f1_residual"].to_numpy(),
        subgroups=subgroups,
    )

    out_json = out_dir / "variance_decomposition.json"
    out_json.write_text(json.dumps(decomposition, indent=2, default=float))
    print(f"\n[variance_decomposition] wrote {out_json}")

    _print_summary(decomposition)
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Variance decomposition for ResDec-H3 composite predictions.",
    )
    p.add_argument(
        "--pred-root", default="outputs/redesign/p5_canonical_seed42",
        help="Directory containing fold{0..4}/val_predictions_best.npz",
    )
    p.add_argument(
        "--tabpfn-dir", default="data/redesign",
        help="Directory containing tabpfn_outer_fold{0..4}.npz",
    )
    p.add_argument(
        "--metadata-csv", default="data/metadata_ROSMAP/metadata.csv",
        help="ROSMAP metadata CSV with apoe_genotype / msex / age_death columns",
    )
    p.add_argument(
        "--out-dir", default="outputs/redesign/interpretability",
        help="Output directory (will be created if missing)",
    )
    sys.exit(main(p.parse_args()))
