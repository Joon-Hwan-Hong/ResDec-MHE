"""Subgroup stratified metrics for ResDec-H3 composite predictions.

Loads per-fold val predictions (composite ``y_hat = y_tabpfn + f_1``), joins
with ROSMAP metadata, and computes per-subgroup R², RMSE, Pearson r, and
Spearman ρ of the composite vs ``y_true`` with 95% percentile-bootstrap CIs.

Subgroup families (flat group names; see ``subgroup_family`` column in the
CSV output for grouping-back downstream):

- APOE-ε4 count: ``APOE_e4_{0,1,2}`` from ``apoe_genotype`` (count of "4"s)
- Sex: ``msex_{0,1}`` from ``msex``
- Age quartile: ``age_quartile_{Q1..Q4}`` from ``age_death``
- Pathology quartile: ``pathology_quartile_{Q1..Q4}`` from ``gpath``

Usage
-----
    PYTHONPATH=<worktree-root> \\
    uv run python scripts/redesign/interpretability/subgroup_r2.py \\
        --pred-root outputs/redesign/p5_canonical_seed42 \\
        --tabpfn-dir data/redesign \\
        --metadata-csv data/metadata_ROSMAP/metadata.csv \\
        --out-dir outputs/redesign/interpretability

Outputs ``<out-dir>/subgroup_metrics.json`` + ``<out-dir>/subgroup_metrics_table.csv``.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
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

# Reuse the per-fold loader from the shared resdec_io module so both analyses
# see the same subject set. Stratification labels come from the shared helper
# module so both scripts bucket subjects identically.
from src.analysis.resdec_io import load_all_folds  # noqa: E402
from src.analysis.resdec_subgroup_analysis import stratified_metrics  # noqa: E402
from src.analysis.subgroup_helpers import (  # noqa: E402
    apoe_e4_count_label,
    msex_label,
    quantile_labels,
)

logger = logging.getLogger(__name__)


# Metadata columns pulled for the join. Keep explicit so a rename in the
# source CSV surfaces as a KeyError rather than silently skipping a subgroup.
_METADATA_COLS = ["ROSMAP_IndividualID", "apoe_genotype", "msex", "age_death", "gpath"]

# Subgroups with fewer than this many subjects yield bootstrap CIs that are
# too wide to be informative (<10 subjects → percentile CI dominated by
# resample noise, not data variation). We still compute and report metrics
# for them, but the orchestration emits an explicit warning so downstream
# readers know to discount those intervals.
MIN_N_FOR_CI_TRUST = 10


def _build_flat_masks(df: pd.DataFrame) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """Build the flat ``{group_name: boolean_mask[N]}`` dict for all subgroup families.

    Returns a tuple ``(masks, family_by_group)`` where ``family_by_group`` maps
    each flat group name to its subgroup family (``"APOE_e4"``, ``"msex"``,
    ``"age_quartile"``, ``"pathology_quartile"``) for CSV export.
    """
    n = len(df)
    masks: dict[str, np.ndarray] = {}
    family_by_group: dict[str, str] = {}

    # --- APOE-ε4 count --------------------------------------------------
    apoe_labels = df["apoe_genotype"].apply(apoe_e4_count_label)
    for count in ("0", "1", "2"):
        group_name = f"APOE_e4_{count}"
        masks[group_name] = (apoe_labels == count).to_numpy(dtype=bool)
        family_by_group[group_name] = "APOE_e4"

    # --- Sex ------------------------------------------------------------
    msex_labels = df["msex"].apply(msex_label)
    for sex in ("0", "1"):
        group_name = f"msex_{sex}"
        masks[group_name] = (msex_labels == sex).to_numpy(dtype=bool)
        family_by_group[group_name] = "msex"

    # --- Age quartile ---------------------------------------------------
    age_q = quantile_labels(df["age_death"], n_quantiles=4, prefix="Q")
    for q in ("Q1", "Q2", "Q3", "Q4"):
        group_name = f"age_quartile_{q}"
        masks[group_name] = (age_q == q).to_numpy(dtype=bool)
        family_by_group[group_name] = "age_quartile"

    # --- Pathology quartile ---------------------------------------------
    gpath_q = quantile_labels(df["gpath"], n_quantiles=4, prefix="Q")
    for q in ("Q1", "Q2", "Q3", "Q4"):
        group_name = f"pathology_quartile_{q}"
        masks[group_name] = (gpath_q == q).to_numpy(dtype=bool)
        family_by_group[group_name] = "pathology_quartile"

    # Sanity: every mask must have length N.
    for k, m in masks.items():
        if m.shape != (n,):
            raise RuntimeError(
                f"Mask for {k!r} has shape {m.shape}; expected ({n},)."
            )

    return masks, family_by_group


def _metrics_to_rows(
    metrics: dict[str, dict],
    family_by_group: dict[str, str],
) -> list[dict]:
    """Flatten the nested metrics dict to CSV-ready rows.

    One row per group. Columns: ``subgroup_family, group_label, n,
    r2, r2_ci_low, r2_ci_high, rmse, rmse_ci_low, rmse_ci_high, pearson_r,
    pearson_r_ci_low, pearson_r_ci_high, spearman_rho, spearman_rho_ci_low,
    spearman_rho_ci_high, n_valid_bootstraps``.
    """
    rows: list[dict] = []
    for group_name, stats in metrics.items():
        row = {
            "subgroup_family": family_by_group.get(group_name, ""),
            "group_label": group_name,
            "n": stats["n"],
            "r2": stats["r2"],
            "r2_ci_low": stats["r2_ci"][0],
            "r2_ci_high": stats["r2_ci"][1],
            "rmse": stats["rmse"],
            "rmse_ci_low": stats["rmse_ci"][0],
            "rmse_ci_high": stats["rmse_ci"][1],
            "pearson_r": stats["pearson_r"],
            "pearson_r_ci_low": stats["pearson_r_ci"][0],
            "pearson_r_ci_high": stats["pearson_r_ci"][1],
            "spearman_rho": stats["spearman_rho"],
            "spearman_rho_ci_low": stats["spearman_rho_ci"][0],
            "spearman_rho_ci_high": stats["spearman_rho_ci"][1],
            "n_valid_bootstraps": stats["n_valid_bootstraps"],
        }
        rows.append(row)
    return rows


def _print_summary(metrics: dict[str, dict], family_by_group: dict[str, str]) -> None:
    """Human-readable stdout summary: per-family table of n, R², R² CI."""
    # Group by family, preserve insertion order within each family.
    by_family: dict[str, list[str]] = {}
    for group_name, fam in family_by_group.items():
        by_family.setdefault(fam, []).append(group_name)

    print("\n=== Subgroup stratified metrics ===")
    for fam, groups in by_family.items():
        print(f"\n  --- {fam} ---")
        print(f"    {'group':<28s}  {'n':>5s}  {'R²':>8s}  {'R² 95% CI':>22s}")
        for g in groups:
            s = metrics[g]
            r2 = s["r2"]
            lo, hi = s["r2_ci"]
            r2_s = f"{r2:8.4f}" if not math.isnan(r2) else "     nan"
            ci_s = (
                f"[{lo:7.4f}, {hi:7.4f}]"
                if not (math.isnan(lo) or math.isnan(hi))
                else "[    nan,     nan]"
            )
            print(f"    {g:<28s}  {s['n']:5d}  {r2_s}  {ci_s:>22s}")


def main(args: argparse.Namespace) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    pred_root = Path(args.pred_root)
    tabpfn_dir = Path(args.tabpfn_dir)
    metadata_csv = Path(args.metadata_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("[subgroup_r2] pred-root     = %s", pred_root)
    logger.info("[subgroup_r2] tabpfn-dir    = %s", tabpfn_dir)
    logger.info("[subgroup_r2] metadata      = %s", metadata_csv)
    logger.info("[subgroup_r2] out-dir       = %s", out_dir)
    logger.info("[subgroup_r2] n-folds       = %d", args.n_folds)
    logger.info("[subgroup_r2] n-bootstrap   = %d", args.n_bootstrap)
    logger.info("[subgroup_r2] seed          = %d", args.seed)

    # 1. Load concatenated per-fold val predictions via the C.1 loader.
    df_pred = load_all_folds(pred_root, tabpfn_dir, n_folds=args.n_folds)
    logger.info(
        "[subgroup_r2] loaded %d subjects across %d folds",
        len(df_pred), df_pred["fold"].nunique(),
    )

    # 2. Metadata join.
    if not metadata_csv.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {metadata_csv}")
    meta = pd.read_csv(metadata_csv)
    for col in _METADATA_COLS:
        if col not in meta.columns:
            raise KeyError(f"metadata.csv missing required column: {col!r}")

    df = df_pred.merge(meta[_METADATA_COLS], on="ROSMAP_IndividualID", how="left")
    logger.info(
        "[subgroup_r2] metadata join: "
        "APOE available for %d/%d, msex for %d/%d, "
        "age_death for %d/%d, gpath for %d/%d",
        df["apoe_genotype"].notna().sum(), len(df),
        df["msex"].notna().sum(), len(df),
        df["age_death"].notna().sum(), len(df),
        df["gpath"].notna().sum(), len(df),
    )

    # 3. Build flat boolean masks for every subgroup family.
    masks, family_by_group = _build_flat_masks(df)
    # Sanity: masks and family_by_group must cover the exact same set of
    # group names. A mismatch indicates a refactor drift (e.g., a new family
    # added to one dict but not the other).
    assert set(masks.keys()) == set(family_by_group.keys()), (
        f"mask/family key drift: "
        f"{set(masks.keys()) ^ set(family_by_group.keys())}"
    )
    for group_name, mask in masks.items():
        logger.info(
            "[subgroup_r2] %s: n=%d (family=%s)",
            group_name, int(mask.sum()), family_by_group[group_name],
        )

    # 4. Compute stratified metrics + bootstrap CIs.
    metrics = stratified_metrics(
        df["y_true"].to_numpy(),
        df["y_composite"].to_numpy(),
        masks,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )

    # 4b. Flag subgroups too small for trustworthy CIs. The metrics are still
    # computed and written to disk, but the reader should treat the bootstrap
    # interval as dominated by resample noise rather than data variation.
    for group_name, stats in metrics.items():
        if stats["n"] < MIN_N_FOR_CI_TRUST:
            logger.warning(
                "[subgroup_r2] %s: n=%d is below reliability threshold "
                "(n<%d); R²=%.3f CI=%s is highly uncertain",
                group_name, stats["n"], MIN_N_FOR_CI_TRUST,
                stats["r2"], stats["r2_ci"],
            )

    # 5. Write JSON + CSV.
    out_json = out_dir / "subgroup_metrics.json"
    out_json.write_text(json.dumps(metrics, indent=2, default=float))
    logger.info("[subgroup_r2] wrote %s", out_json)

    rows = _metrics_to_rows(metrics, family_by_group)
    out_csv = out_dir / "subgroup_metrics_table.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    logger.info("[subgroup_r2] wrote %s", out_csv)

    # 6. Readable summary to stdout.
    _print_summary(metrics, family_by_group)
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Subgroup stratified metrics for ResDec-H3 composite predictions.",
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
        help="ROSMAP metadata CSV with apoe_genotype / msex / age_death / gpath columns",
    )
    p.add_argument(
        "--out-dir", default="outputs/redesign/interpretability",
        help="Output directory (will be created if missing)",
    )
    p.add_argument(
        "--n-folds", type=int, default=5,
        help="Number of outer folds (default: 5).",
    )
    p.add_argument(
        "--n-bootstrap", type=int, default=1000,
        help="Bootstrap resamples per subgroup (default: 1000).",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Seed for np.random.default_rng (default: 42).",
    )
    sys.exit(main(p.parse_args()))
