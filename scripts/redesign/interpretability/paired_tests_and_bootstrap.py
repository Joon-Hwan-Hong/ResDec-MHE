"""Statistical rigor orchestration for ResDec-H3 composite predictions.

Loads per-fold R² for "ours" (ResDec-H3 composite) and every discovered
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
    uv run python scripts/redesign/interpretability/paired_tests_and_bootstrap.py \\
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
from sklearn.metrics import r2_score

# Make the script standalone-runnable: ensure the worktree root is on sys.path.
_WORKTREE_ROOT = Path(__file__).resolve().parents[3]
if not (_WORKTREE_ROOT / "src").is_dir():
    raise RuntimeError(
        f"sys.path bootstrap failed: {_WORKTREE_ROOT}/src not found; "
        "set PYTHONPATH=<worktree-root>."
    )
sys.path.insert(0, str(_WORKTREE_ROOT))

# Reuse the per-fold loader from C.1 so ours composite predictions come
# from the same subject set every interpretability script sees.
from scripts.redesign.interpretability.variance_decomposition import (  # noqa: E402
    load_all_folds,
)
from src.analysis.resdec_statistical_rigor import (  # noqa: E402
    bootstrap_r2_ci,
    calibration_coverage,
    paired_wilcoxon,
)

logger = logging.getLogger(__name__)


# Nominal coverage levels to report in the calibration block.
_NOMINAL_COVERAGE = [0.5, 0.68, 0.8, 0.95]


def compute_per_fold_r2_ours(df: pd.DataFrame, n_folds: int) -> np.ndarray:
    """Per-fold R²(y_true, y_composite) on our concatenated predictions."""
    r2s = np.empty(n_folds, dtype=np.float64)
    for f in range(n_folds):
        sub = df[df["fold"] == f]
        if len(sub) == 0:
            raise RuntimeError(f"Ours: fold {f} has 0 subjects in df.")
        r2s[f] = r2_score(sub["y_true"].to_numpy(), sub["y_composite"].to_numpy())
    return r2s


def compute_per_fold_r2_tabpfn(tabpfn_dir: Path, n_folds: int) -> np.ndarray:
    """Per-fold R²(y_true, y_tabpfn) from ``tabpfn_outer_fold{f}.npz``.

    Fails loud if any fold's npz is missing — TabPFN-2.6 standalone is the
    required strongest baseline for this study and cannot be skipped.
    """
    r2s = np.empty(n_folds, dtype=np.float64)
    for f in range(n_folds):
        path = tabpfn_dir / f"tabpfn_outer_fold{f}.npz"
        if not path.exists():
            raise FileNotFoundError(
                f"TabPFN-2.6 standalone baseline is required; missing {path}"
            )
        tab = np.load(path, allow_pickle=True)
        r2s[f] = r2_score(
            tab["y_true"].astype(np.float64),
            tab["y_tabpfn"].astype(np.float64),
        )
    return r2s


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
    for name, stats in results["paired_wilcoxon"].items():
        lines.append(
            f"| {name} | {stats['n_folds']} | "
            f"{stats['median_diff']:+.4f} | "
            f"{stats['statistic']:.1f} | "
            f"{stats['p_value']:.4f} |"
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
    lines.append(f"_Uncertainty proxy: TabPFN-2.6 per-subject `sigma_tabpfn`_")
    lines.append(f"_(composite head has no independent σ; this is the best-available proxy)_")
    lines.append("")
    lines.append("| Nominal | Empirical coverage |")
    lines.append("|---------|--------------------|")
    for p in _NOMINAL_COVERAGE:
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


def _print_summary(results: dict) -> None:
    """Stdout summary: paired-Wilcoxon table, bootstrap CI, coverage."""
    print("\n=== Paired Wilcoxon (per-fold R², ours vs baseline, alt='greater') ===")
    print(f"  {'baseline':<28s}  {'n':>2s}  {'med ΔR²':>10s}  {'W':>8s}  {'p':>10s}")
    for name, stats in results["paired_wilcoxon"].items():
        print(
            f"  {name:<28s}  {stats['n_folds']:>2d}  "
            f"{stats['median_diff']:>+10.4f}  "
            f"{stats['statistic']:>8.1f}  "
            f"{stats['p_value']:>10.4f}"
        )

    b = results["bootstrap_r2_ci"]
    print(f"\n=== Bootstrap R² CI (N={b['n']}, n_boot={b['n_boot']}) ===")
    print(f"  point R²:      {b['point_r2']:.4f}")
    print(
        f"  {int(b['conf'] * 100)}% CI: "
        f"[{b['ci_lower']:.4f}, {b['ci_upper']:.4f}]"
    )

    c = results["calibration_coverage"]
    print("\n=== Calibration coverage (σ_tabpfn proxy) ===")
    for p in _NOMINAL_COVERAGE:
        key = f"coverage_at_{p}"
        if key in c:
            print(f"  nominal {int(p * 100):>2d}% → empirical {c[key]:.3f}")
    print(f"  mean σ_tabpfn:       {c['mean_sigma']:.4f}")
    print(f"  mean |residual|:     {c['mean_abs_residual']:.4f}")
    print(f"  n: {c['n']}")


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

    logger.info("[paired_tests] pred-root        = %s", pred_root)
    logger.info("[paired_tests] tabpfn-dir       = %s", tabpfn_dir)
    logger.info("[paired_tests] baselines-root   = %s", baselines_root)
    logger.info("[paired_tests] out-dir          = %s", out_dir)
    logger.info("[paired_tests] n-folds          = %d", args.n_folds)
    logger.info("[paired_tests] n-boot           = %d", args.n_boot)
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
    paired = {"TabPFN-2.6_standalone": paired_wilcoxon(
        r2s_ours, r2s_tabpfn, alternative="greater",
    )}
    for name, r2s_bl in sorted(other_baselines.items()):
        paired[name] = paired_wilcoxon(
            r2s_ours, r2s_bl, alternative="greater",
        )

    # 6. Bootstrap CI on pooled composite predictions.
    boot = bootstrap_r2_ci(
        df_ours["y_true"].to_numpy(),
        df_ours["y_composite"].to_numpy(),
        n_boot=args.n_boot,
        conf=0.95,
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
        nominal=_NOMINAL_COVERAGE,
    )

    # 8. Assemble results + write JSON + MD.
    results = {
        "paired_wilcoxon": paired,
        "bootstrap_r2_ci": boot,
        "calibration_coverage": calib,
        "provenance": {
            "pred_root": str(pred_root),
            "tabpfn_dir": str(tabpfn_dir),
            "baselines_root": str(baselines_root),
            "n_folds": int(args.n_folds),
            "n_boot": int(args.n_boot),
            "seed": int(args.seed),
            "n_subjects": int(len(df_ours)),
            "sigma_source_note": (
                "Calibration coverage computed using TabPFN-2.6 "
                "per-subject sigma_tabpfn (from tabpfn_outer_fold{f}.npz) "
                "as a proxy for composite predictive uncertainty. The "
                "composite head has no independent sigma; this is the "
                "best-available proxy."
            ),
            "per_fold_r2": {
                "ours": [float(v) for v in r2s_ours],
                "TabPFN-2.6_standalone": [float(v) for v in r2s_tabpfn],
                **{k: [float(v) for v in r2s] for k, r2s in other_baselines.items()},
            },
        },
    }

    out_json = out_dir / "statistical_rigor.json"
    out_json.write_text(json.dumps(results, indent=2, default=float))
    logger.info("[paired_tests] wrote %s", out_json)

    out_md = out_dir / "statistical_rigor.md"
    out_md.write_text(build_markdown_report(results))
    logger.info("[paired_tests] wrote %s", out_md)

    _print_summary(results)
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Statistical rigor for ResDec-H3 composite predictions: "
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
        "--seed", type=int, default=42,
        help="Seed for np.random.default_rng (default: 42).",
    )
    sys.exit(main(p.parse_args()))
