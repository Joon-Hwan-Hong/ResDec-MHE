"""Variance decomposition of the ResDec-MHE composite prediction.

Loads per-fold val predictions (composite ``y_hat = y_tabpfn + f_1``) and the
frozen outer-fold TabPFN-2.6 predictions, recovers the neural-head residual
``f_1 = y_hat - y_tabpfn`` per subject, joins with ROSMAP metadata, and writes
a JSON report with the variance budget decomposed as

    Var(y) = Var(y_tabpfn) + Var(f_1) + 2 Cov(y_tabpfn, f_1) + Var(resid)

globally and stratified by APOE-ε4 count, sex, and age-at-death quartile.

Usage
-----
    PYTHONPATH=<worktree-root> \\
    uv run python scripts/resdec_mhe/interpretability/variance_decomposition.py \\
        --pred-root outputs/redesign/p5_canonical_seed42 \\
        --tabpfn-dir data/redesign \\
        --metadata-csv data/metadata_ROSMAP/metadata.csv \\
        --out-dir outputs/redesign/interpretability

Outputs ``<out-dir>/variance_decomposition.json``.
"""
from __future__ import annotations

import argparse
import json
import logging
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

from src.analysis.resdec_io import load_all_folds
from src.analysis.resdec_variance_decomposition import decompose_variance
from src.analysis.subgroup_helpers import (
    apoe_e4_count_label,
    msex_label,
    quantile_labels,
)

logger = logging.getLogger(__name__)


def _build_subgroups(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Build the three stratifications required by the spec."""
    # APOE-ε4 count → "0" / "1" / "2"; None when APOE genotype is missing.
    apoe_str = df["apoe_genotype"].apply(apoe_e4_count_label)

    # Sex: msex ∈ {0, 1}. Keep NaN → None.
    msex_str = df["msex"].apply(msex_label)

    age_q = quantile_labels(df["age_death"], n_quantiles=4, prefix="Q")

    return {
        "by_apoe_e4_count": apoe_str.to_numpy(dtype=object),
        "by_msex": msex_str.to_numpy(dtype=object),
        "by_age_quartile": age_q.to_numpy(dtype=object),
    }


def _print_summary(decomposition: dict) -> None:
    """Human-readable stdout summary (global fractions + per-subgroup counts).

    When ``var_y == 0`` (degenerate target), the fraction-of-Var(y) columns
    divide by NaN so they render as "nan%" — still valid output, and the
    upstream decomposition also returns NaN for ``total_explained_fraction``.
    """
    g = decomposition["global"]
    var_y = g["var_y"]
    denom = var_y if var_y > 0 else float("nan")
    print("\n=== Variance Decomposition (global) ===")
    print(f"  n                      : {g['n']}")
    print(f"  Var(y)                 : {g['var_y']:.4f}")
    print(f"  Var(y_tabpfn)          : {g['var_tabpfn']:.4f}  "
          f"({g['var_tabpfn'] / denom:.1%} of Var(y))")
    print(f"  Var(f_1)               : {g['var_f1']:.4f}  "
          f"({g['var_f1'] / denom:.1%} of Var(y))")
    print(f"  2 Cov(y_tabpfn, f_1)   : {2 * g['cov_tabpfn_f1']:+.4f}  "
          f"({2 * g['cov_tabpfn_f1'] / denom:+.1%} of Var(y))")
    print(f"  Var(resid)             : {g['var_resid']:.4f}  "
          f"({g['var_resid'] / denom:.1%} of Var(y))")
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
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    pred_root = Path(args.pred_root)
    tabpfn_dir = Path(args.tabpfn_dir)
    metadata_csv = Path(args.metadata_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[variance_decomposition] pred-root   = {pred_root}")
    print(f"[variance_decomposition] tabpfn-dir  = {tabpfn_dir}")
    print(f"[variance_decomposition] metadata    = {metadata_csv}")
    print(f"[variance_decomposition] out-dir     = {out_dir}")
    print(f"[variance_decomposition] n-folds     = {args.n_folds}")

    df_pred = load_all_folds(pred_root, tabpfn_dir, n_folds=args.n_folds)
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
    # Explicit visibility on subjects silently dropped for missing metadata.
    for key, labels in subgroups.items():
        missing = sum(
            1 for lbl in labels
            if lbl is None or (isinstance(lbl, float) and np.isnan(lbl))
        )
        n_labeled = len(labels) - missing
        logger.info(
            "[variance_decomposition] %s: %d/%d subjects labeled "
            "(%d dropped - missing metadata)",
            key, n_labeled, len(labels), missing,
        )

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
        description="Variance decomposition for ResDec-MHE composite predictions.",
    )
    p.add_argument(
        "--pred-root", default="outputs/redesign/p5_canonical_seed42",
        help="Directory containing fold{0..N-1}/val_predictions_best.npz",
    )
    p.add_argument(
        "--tabpfn-dir", default="data/redesign",
        help="Directory containing tabpfn_outer_fold{0..N-1}.npz",
    )
    p.add_argument(
        "--metadata-csv", default="data/metadata_ROSMAP/metadata.csv",
        help="ROSMAP metadata CSV with apoe_genotype / msex / age_death columns",
    )
    p.add_argument(
        "--out-dir", default="outputs/redesign/interpretability",
        help="Output directory (will be created if missing)",
    )
    p.add_argument(
        "--n-folds", type=int, default=5,
        help="Number of outer folds (default: 5).",
    )
    sys.exit(main(p.parse_args()))
