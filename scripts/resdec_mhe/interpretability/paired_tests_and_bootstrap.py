"""Statistical rigor orchestration for ResDec-MHE composite predictions.

Loads per-fold R² for "ours" (ResDec-MHE composite) and every discovered
baseline, runs paired one-sided Wilcoxon signed-rank tests (ours vs each
baseline, ``alternative="greater"``), computes a percentile bootstrap CI
on the pooled-N composite R² (resample 516 subjects 1000×), and reports
empirical calibration coverage at nominal levels
``{0.5, 0.68, 0.8, 0.95}`` using TabPFN-2.6's per-subject ``sigma_tabpfn``
as a proxy for composite predictive uncertainty (documented in output).

Baseline discovery
------------------
1. **TabPFN-2.6 standalone (required)**: per-fold R² computed on the fly
   from ``data/redesign/tabpfn_outer_fold{0..4}.npz`` using
   ``sklearn.metrics.r2_score(y_true, y_tabpfn)``. If ANY fold's npz is
   missing, fail loud.
2. **Other baselines (optional)**: glob ``<baselines-root>/*/results.csv``;
   parse columns ``r2`` and ``fold``; include only if 5 per-fold values
   are present. Missing baselines emit a WARNING and are skipped.

Usage
-----
    PYTHONPATH=<worktree-root> \\
    uv run python scripts/resdec_mhe/interpretability/paired_tests_and_bootstrap.py \\
        --pred-root outputs/redesign/p5_canonical_seed42 \\
        --tabpfn-dir data/redesign \\
        --baselines-root outputs/baselines \\
        --out-dir outputs/redesign/interpretability \\
        --n-boot 1000 --seed 42

Outputs:
- ``<out-dir>/statistical_rigor.json`` — full dict of results.
- ``<out-dir>/statistical_rigor.md`` — paper-ready markdown table.
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

# Reuse the shared resdec_io loaders so ours composite predictions come from
# the same subject set every interpretability script sees. compute_per_fold_r2_*
# helpers also live in the shared module for reuse across the baseline table etc.
from src.analysis.resdec_io import (
    compute_per_fold_r2_ours,
    compute_per_fold_r2_tabpfn,
    load_all_folds,
)
from src.analysis.resdec_statistical_rigor import (
    bootstrap_r2_ci,
    calibration_coverage,
    paired_wilcoxon,
)

logger = logging.getLogger(__name__)


# Nominal coverage levels to report in the calibration block. 0.68 here is the
# rounded "68%" label used by paper tables — NOT the 1-σ convention 0.6827.
_DEFAULT_NOMINAL_COVERAGE: tuple[float, ...] = (0.5, 0.68, 0.8, 0.95)

# Canonical (snake_case) baseline keys used in the JSON output. The
# _DISPLAY_NAMES mapping renders them human-readable in the markdown report.
_TABPFN_STANDALONE_KEY = "tabpfn_2_6_standalone"
_OURS_KEY = "ours"
_DISPLAY_NAMES: dict[str, str] = {
    _OURS_KEY: "ResDec-MHE (ours)",
    _TABPFN_STANDALONE_KEY: "TabPFN-2.6 standalone",
    "cloudpred": "CloudPred",
    "cloudpred_pertype": "CloudPred (per-type)",
    "gpio": "GPIO",
    "perceiver_io": "Perceiver-IO",
}


def _display_name(key: str) -> str:
    """Return a human-readable label for a baseline key, falling back to the key."""
    return _DISPLAY_NAMES.get(key, key)


def discover_baseline_r2s(
    baselines_root: Path, n_folds: int,
) -> dict[str, np.ndarray]:
    """Glob ``<baselines-root>/*/results.csv``, parse per-fold R² arrays.

    Parameters
    ----------
    baselines_root : Path
        Directory holding per-baseline subdirs. Each subdir may contain
        a ``results.csv`` with at least ``r2`` and ``fold`` columns.
    n_folds : int
        Number of folds required. Baselines with fewer per-fold R²
        values are skipped with a WARNING.

    Returns
    -------
    dict
        ``{baseline_name: fold_r2s[n_folds]}`` (baseline_name = subdir name).
    """
    out: dict[str, np.ndarray] = {}
    if not baselines_root.exists():
        logger.warning(
            "Baselines root %s does not exist; no extra baselines loaded.",
            baselines_root,
        )
        return out

    for subdir in sorted(p for p in baselines_root.iterdir() if p.is_dir()):
        csv_path = subdir / "results.csv"
        if not csv_path.exists():
            logger.warning(
                "Baseline %s: no results.csv at %s; skipping.",
                subdir.name, csv_path,
            )
            continue
        try:
            df = pd.read_csv(csv_path)
        except Exception as exc:
            logger.warning(
                "Baseline %s: failed to read %s (%s); skipping.",
                subdir.name, csv_path, exc,
            )
            continue
        if "r2" not in df.columns or "fold" not in df.columns:
            logger.warning(
                "Baseline %s: %s missing 'r2' or 'fold' columns (have %s); skipping.",
                subdir.name, csv_path, list(df.columns),
            )
            continue

        # Fold integrity: the fold column must coerce cleanly to numeric and
        # cover at least n_folds unique values. Non-numeric / sparse folds
        # would silently sort wrong without this guard.
        try:
            df["fold"] = pd.to_numeric(df["fold"], errors="raise")
        except (ValueError, TypeError) as exc:
            logger.warning(
                "Baseline %s: non-numeric 'fold' column (%s); skipping.",
                subdir.name, exc,
            )
            continue
        if df["fold"].nunique() < n_folds:
            logger.warning(
                "Baseline %s: only %d unique folds (expected %d); skipping.",
                subdir.name, df["fold"].nunique(), n_folds,
            )
            continue

        # Per-fold ordering: sort by fold to align with our per-fold R² array.
        # Baseline CSVs may use 1-indexed fold ids; normalise to 0-indexed by
        # sorting and then slicing the first n_folds rows.
        df = df.sort_values("fold").reset_index(drop=True)
        if len(df) < n_folds:
            logger.warning(
                "Baseline %s: only %d folds in %s (need %d); skipping.",
                subdir.name, len(df), csv_path, n_folds,
            )
            continue
        r2s = df["r2"].to_numpy(dtype=np.float64)[:n_folds]
        if not np.all(np.isfinite(r2s)):
            logger.warning(
                "Baseline %s: non-finite R² in %s (values=%s); skipping.",
                subdir.name, csv_path, r2s,
            )
            continue
        out[subdir.name] = r2s
        logger.info(
            "[paired_tests] discovered baseline %s from %s (R² per fold = %s)",
            subdir.name, csv_path,
            ", ".join(f"{v:.4f}" for v in r2s),
        )
    return out


def concat_sigma_tabpfn(tabpfn_dir: Path, n_folds: int) -> pd.DataFrame:
    """Return concat DataFrame of (subject_id, sigma_tabpfn) across folds.

    Downstream joins this onto our composite predictions via ROSMAP_IndividualID.
    """
    frames = []
    for f in range(n_folds):
        path = tabpfn_dir / f"tabpfn_outer_fold{f}.npz"
        tab = np.load(path, allow_pickle=True)
        frames.append(pd.DataFrame({
            "ROSMAP_IndividualID": tab["val_subject_ids"].astype(str),
            "sigma_tabpfn": tab["sigma_tabpfn"].astype(np.float64),
        }))
    return pd.concat(frames, ignore_index=True)


def build_markdown_report(results: dict) -> str:
    """Render the results dict as a paper-ready markdown table."""
    lines: list[str] = []

    # --- Paired Wilcoxon table ---------------------------------------------
    lines.append("## Paired Wilcoxon signed-rank (per-fold R², ours vs baseline, alternative=greater)")
    lines.append("")
    lines.append("| Baseline | n | median ΔR² | W statistic | p-value |")
    lines.append("|----------|---|-----------|-------------|---------|")
    for name, entry in results["paired_wilcoxon"].items():
        lines.append(
            f"| {_display_name(name)} | {entry['n_folds']} | "
            f"{entry['median_diff']:+.4f} | "
            f"{entry['statistic']:.1f} | "
            f"{entry['p_value']:.4f} |"
        )
    lines.append("")

    # --- Bootstrap CI ------------------------------------------------------
    b = results["bootstrap_r2_ci"]
    lines.append(f"## Bootstrap R² CI (our composite predictions, N={b['n']})")
    lines.append("")
    lines.append(f"- Point R²:  {b['point_r2']:.4f}")
    lines.append(
        f"- {int(b['conf'] * 100)}% CI: [{b['ci_lower']:.4f}, {b['ci_upper']:.4f}]"
    )
    lines.append(f"- n_boot:    {b['n_boot']}")
    lines.append("")

    # --- Calibration coverage ---------------------------------------------
    c = results["calibration_coverage"]
    lines.append("## Calibration coverage")
    lines.append("")
    lines.append(
        "> Uncertainty proxy: TabPFN-2.6 per-subject `sigma_tabpfn` "
        "(composite head has no independent σ; this is the best-available proxy)."
    )
    lines.append("")
    lines.append("| Nominal | Empirical coverage |")
    lines.append("|---------|--------------------|")
    nominal_levels = results.get("provenance", {}).get(
        "nominal_coverage", list(_DEFAULT_NOMINAL_COVERAGE),
    )
    for p in nominal_levels:
        key = f"coverage_at_{p}"
        if key in c:
            lines.append(f"| {int(p * 100)}% | {c[key]:.3f} |")
    lines.append("")
    lines.append(
        f"- mean σ_tabpfn:         {c['mean_sigma']:.4f}"
    )
    lines.append(
        f"- mean |y_true − y_hat|: {c['mean_abs_residual']:.4f}"
    )
    lines.append(f"- n: {c['n']}")
    lines.append("")
    return "\n".join(lines)


def main(args: argparse.Namespace) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    pred_root = Path(args.pred_root)
    tabpfn_dir = Path(args.tabpfn_dir)
    baselines_root = Path(args.baselines_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    nominal_coverage = tuple(float(x) for x in args.nominal_coverage)

    logger.info("[paired_tests] pred-root        = %s", pred_root)
    logger.info("[paired_tests] tabpfn-dir       = %s", tabpfn_dir)
    logger.info("[paired_tests] baselines-root   = %s", baselines_root)
    logger.info("[paired_tests] out-dir          = %s", out_dir)
    logger.info("[paired_tests] n-folds          = %d", args.n_folds)
    logger.info("[paired_tests] n-boot           = %d", args.n_boot)
    logger.info("[paired_tests] conf             = %.4f", args.conf)
    logger.info("[paired_tests] nominal-coverage = %s", list(nominal_coverage))
    logger.info("[paired_tests] seed             = %d", args.seed)

    # 1. Our composite predictions (joined over all folds).
    df_ours = load_all_folds(pred_root, tabpfn_dir, n_folds=args.n_folds)
    logger.info(
        "[paired_tests] loaded %d ours subjects across %d folds",
        len(df_ours), df_ours["fold"].nunique(),
    )

    # 2. Per-fold R² — ours.
    r2s_ours = compute_per_fold_r2_ours(df_ours, n_folds=args.n_folds)
    logger.info(
        "[paired_tests] ours per-fold R²: %s (mean=%.4f ± %.4f)",
        ", ".join(f"{v:.4f}" for v in r2s_ours),
        float(r2s_ours.mean()), float(r2s_ours.std(ddof=1)),
    )

    # 3. Per-fold R² — TabPFN-2.6 standalone (required).
    r2s_tabpfn = compute_per_fold_r2_tabpfn(tabpfn_dir, n_folds=args.n_folds)
    logger.info(
        "[paired_tests] TabPFN-2.6 standalone per-fold R²: %s (mean=%.4f ± %.4f)",
        ", ".join(f"{v:.4f}" for v in r2s_tabpfn),
        float(r2s_tabpfn.mean()), float(r2s_tabpfn.std(ddof=1)),
    )

    # 4. Discover other baselines under `<baselines-root>/*/results.csv`.
    other_baselines = discover_baseline_r2s(baselines_root, n_folds=args.n_folds)
    logger.info(
        "[paired_tests] discovered %d other baselines: %s",
        len(other_baselines), sorted(other_baselines.keys()),
    )

    # 5. Paired Wilcoxon: ours vs each baseline.
    paired: dict[str, dict] = {
        _TABPFN_STANDALONE_KEY: paired_wilcoxon(
            r2s_ours, r2s_tabpfn, alternative="greater",
        ),
    }
    for name, r2s_bl in sorted(other_baselines.items()):
        paired[name] = paired_wilcoxon(
            r2s_ours, r2s_bl, alternative="greater",
        )

    # 6. Bootstrap CI on pooled composite predictions.
    boot = bootstrap_r2_ci(
        df_ours["y_true"].to_numpy(),
        df_ours["y_composite"].to_numpy(),
        n_boot=args.n_boot,
        conf=args.conf,
        seed=args.seed,
    )

    # 7. Calibration coverage using TabPFN-2.6 σ_tabpfn as proxy.
    #    We join per-subject σ_tabpfn onto our composite predictions by
    #    ROSMAP_IndividualID. Every fold's val subjects must map 1:1 to the
    #    corresponding outer-fold TabPFN cache.
    sigma_df = concat_sigma_tabpfn(tabpfn_dir, n_folds=args.n_folds)
    merged = df_ours.merge(sigma_df, on="ROSMAP_IndividualID", how="inner")
    if len(merged) != len(df_ours):
        raise RuntimeError(
            "[paired_tests] sigma_tabpfn join dropped subjects: "
            f"{len(df_ours)} in ours, {len(merged)} after join. "
            "Outer-fold TabPFN cache is missing subjects."
        )
    calib = calibration_coverage(
        merged["y_true"].to_numpy(),
        merged["y_composite"].to_numpy(),
        merged["sigma_tabpfn"].to_numpy(),
        nominal=nominal_coverage,
    )

    # 8. Assemble results + write JSON + MD.
    per_fold_r2: dict[str, list[float]] = {
        _OURS_KEY: [float(v) for v in r2s_ours],
        _TABPFN_STANDALONE_KEY: [float(v) for v in r2s_tabpfn],
    }
    for k, r2s in other_baselines.items():
        per_fold_r2[k] = [float(v) for v in r2s]

    results = {
        "paired_wilcoxon": paired,
        "bootstrap_r2_ci": boot,
        "calibration_coverage": calib,
        "baseline_display_names": {k: _display_name(k) for k in per_fold_r2},
        "provenance": {
            "pred_root": str(pred_root),
            "tabpfn_dir": str(tabpfn_dir),
            "baselines_root": str(baselines_root),
            "n_folds": int(args.n_folds),
            "n_boot": int(args.n_boot),
            "conf": float(args.conf),
            "nominal_coverage": list(nominal_coverage),
            "seed": int(args.seed),
            "n_subjects": int(len(df_ours)),
            "sigma_source_note": (
                "Calibration coverage computed using TabPFN-2.6 "
                "per-subject sigma_tabpfn (from tabpfn_outer_fold{f}.npz) "
                "as a proxy for composite predictive uncertainty. The "
                "composite head has no independent sigma; this is the "
                "best-available proxy."
            ),
            "per_fold_r2": per_fold_r2,
        },
    }

    out_json = out_dir / "statistical_rigor.json"
    out_json.write_text(json.dumps(results, indent=2))
    logger.info("[paired_tests] wrote %s", out_json)

    md_report = build_markdown_report(results)
    out_md = out_dir / "statistical_rigor.md"
    out_md.write_text(md_report)
    logger.info("[paired_tests] wrote %s", out_md)

    # Single summary path: the markdown report is what gets persisted to disk
    # AND echoed to stdout, so terminal output and the on-disk file never drift.
    print(md_report)
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Statistical rigor for ResDec-MHE composite predictions: "
                    "paired Wilcoxon, bootstrap R² CI, calibration coverage.",
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
        "--baselines-root", default="outputs/baselines",
        help="Root dir of baseline subdirs, each with a results.csv "
             "(columns: r2, fold).",
    )
    p.add_argument(
        "--out-dir", default="outputs/redesign/interpretability",
        help="Output directory (will be created if missing).",
    )
    p.add_argument(
        "--n-folds", type=int, default=5,
        help="Number of outer folds (default: 5).",
    )
    p.add_argument(
        "--n-boot", type=int, default=1000,
        help="Bootstrap resamples for global R² CI (default: 1000).",
    )
    p.add_argument(
        "--conf", type=float, default=0.95,
        help="Bootstrap CI confidence level (default: 0.95).",
    )
    p.add_argument(
        "--nominal-coverage", type=float, nargs="+",
        default=list(_DEFAULT_NOMINAL_COVERAGE),
        help="Nominal calibration-coverage levels (default: 0.5 0.68 0.8 0.95).",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Seed for np.random.default_rng (default: 42).",
    )
    sys.exit(main(p.parse_args()))
