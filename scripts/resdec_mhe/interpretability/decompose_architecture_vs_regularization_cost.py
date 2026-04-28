"""Decompose canonical-vs-sweep R² gap into architecture cost vs regularization cost.

Background
----------
The encoder-regularization sweep (``configs/resdec_mhe/entropy_reg.yaml``) at
``--reg-weight 0`` runs the in-graph einsum attention path with the regularizer
multiplied by zero (effectively no regularizer). This means the sweep λ=0 result
IS the einsum-architecture-no-regularizer baseline. The canonical (R²≈0.4436) -
sweep λ=0 (R²≈0.3838) ≈ 0.06 R² is therefore the architecture cost (einsum path
vs SDPA path), independent of regularization. The within-sweep λ-dependent
decrease (λ=0 → λ=1.0: ≈0.3838 → ≈0.3537, Δ≈-0.030) is the pure
regularization-induced cost.

This decomposition lets the paper say: "We bounded the architecture cost at ~0.06
R²; within that architecture, regularization further reduces R² by ~0.030 at the
highest λ tested."

Reads
-----
- ``outputs/redesign/p5_canonical_seed42/fold{0..4}/best_summary.json``
- ``outputs/redesign/p5_entropy_reg/lambda_*/fold{0..4}/summary.json``
- ``outputs/redesign/p5_diff_test/fold{0..4}/summary.json``

Writes
------
``outputs/redesign/interpretability/architecture_vs_regularization_decomposition.json``
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import wilcoxon

from src.utils.provenance import git_sha

logger = logging.getLogger(__name__)


# Lambda directory names → numeric values (kept in deterministic order).
_LAMBDA_DIR_TO_VALUE: list[tuple[str, float]] = [
    ("lambda_0", 0.0),
    ("lambda_0p001", 0.001),
    ("lambda_0p01", 0.01),
    ("lambda_0p1", 0.1),
    ("lambda_1p0", 1.0),
]
_N_FOLDS = 5


def _read_canonical_per_fold(canonical_root: Path) -> list[float]:
    """Read canonical per-fold R² from ``best_summary.json``."""
    r2s: list[float] = []
    for f in range(_N_FOLDS):
        path = canonical_root / f"fold{f}" / "best_summary.json"
        if not path.exists():
            raise FileNotFoundError(f"Canonical summary missing: {path}")
        d = json.loads(path.read_text())
        # best_summary.json: prefer ckpt_filename_r2 (the in-name R²), fall back
        # to val_results[0]['val/r2']. They are identical for canonical runs.
        if "ckpt_filename_r2" in d:
            r2s.append(float(d["ckpt_filename_r2"]))
        else:
            r2s.append(float(d["val_results"][0]["val/r2"]))
    return r2s


def _read_sweep_per_fold(sweep_root: Path, lambda_dir: str) -> list[float]:
    """Read sweep per-fold R² from ``summary.json``."""
    r2s: list[float] = []
    for f in range(_N_FOLDS):
        path = sweep_root / lambda_dir / f"fold{f}" / "summary.json"
        if not path.exists():
            raise FileNotFoundError(f"Sweep summary missing: {path}")
        d = json.loads(path.read_text())
        r2s.append(float(d["val_results"][0]["val/r2"]))
    return r2s


def _read_diff_test_per_fold(diff_test_root: Path) -> list[float]:
    """Read diff-test per-fold R² from ``summary.json``."""
    r2s: list[float] = []
    for f in range(_N_FOLDS):
        path = diff_test_root / f"fold{f}" / "summary.json"
        if not path.exists():
            raise FileNotFoundError(f"Diff-test summary missing: {path}")
        d = json.loads(path.read_text())
        r2s.append(float(d["val_results"][0]["val/r2"]))
    return r2s


def _summarize(values: list[float]) -> dict[str, Any]:
    """Compute mean, std (ddof=1), median, 25/75 quantiles + raw per_fold list."""
    arr = np.asarray(values, dtype=float)
    return {
        "per_fold": [float(v) for v in values],
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)),
        "median": float(np.median(arr)),
        "q25": float(np.quantile(arr, 0.25)),
        "q75": float(np.quantile(arr, 0.75)),
    }


def _paired_diff_block(
    a: list[float], b: list[float], wilcoxon_alternative: str = "less"
) -> dict[str, Any]:
    """Return summary of paired (a[i] - b[i]) differences plus a one-sided
    Wilcoxon signed-rank test of whether a − b > 0.

    The test uses ``alternative="less"`` on the (b − a) differences, which is
    equivalent to the one-sided alternative ``a > b`` (i.e. the difference
    ``a − b`` is positive). This pattern is conventional for "is X > Y" paired
    tests against the null of zero median difference.
    """
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    diff = a_arr - b_arr
    summary = _summarize([float(v) for v in diff])
    # Run Wilcoxon on (b - a) so that "less" tests b − a < 0 ⇔ a > b.
    try:
        result = wilcoxon(b_arr - a_arr, alternative=wilcoxon_alternative)
        summary["wilcoxon_W"] = float(result.statistic)
        summary["wilcoxon_p_one_sided"] = float(result.pvalue)
    except ValueError as exc:
        # Wilcoxon raises if all differences are zero; surface it explicitly.
        logger.warning("Wilcoxon failed: %s", exc)
        summary["wilcoxon_W"] = float("nan")
        summary["wilcoxon_p_one_sided"] = float("nan")
    summary["wilcoxon_alternative"] = wilcoxon_alternative
    summary["wilcoxon_note"] = (
        "alternative='less' is applied to (b - a); equivalent to one-sided "
        "test of a > b (i.e. a - b > 0)."
    )
    return summary


def _build_interpretation(
    canonical_mean: float,
    canonical_std: float,
    sweep_l0_mean: float,
    sweep_lmax_mean: float,
    diff_test_mean: float,
    arch_cost: dict[str, Any],
    reg_cost: dict[str, Any],
    diff_test_cost: dict[str, Any],
) -> str:
    arch_explains_diff = (
        abs(arch_cost["mean"] - diff_test_cost["mean"]) <= max(
            arch_cost["std"], diff_test_cost["std"]
        )
    )
    return (
        f"Canonical R² (mean ± std, ddof=1): {canonical_mean:.4f} ± "
        f"{canonical_std:.4f}. Sweep λ=0 R²: {sweep_l0_mean:.4f}. Sweep λ=1.0 "
        f"R²: {sweep_lmax_mean:.4f}. Diff-test R²: {diff_test_mean:.4f}. "
        f"Architecture cost (canonical − sweep λ=0, paired per-fold): "
        f"{arch_cost['mean']:.4f} ± {arch_cost['std']:.4f}, Wilcoxon "
        f"(alternative='less' on neg-diff) p_one_sided="
        f"{arch_cost['wilcoxon_p_one_sided']:.4g}. "
        f"Regularization cost at λ=1.0 (sweep λ=0 − sweep λ=1.0, paired): "
        f"{reg_cost['mean']:.4f} ± {reg_cost['std']:.4f}, Wilcoxon "
        f"p_one_sided={reg_cost['wilcoxon_p_one_sided']:.4g}. "
        f"Diff-test cost (canonical − diff-test, paired): "
        f"{diff_test_cost['mean']:.4f} ± {diff_test_cost['std']:.4f}, "
        f"Wilcoxon p_one_sided={diff_test_cost['wilcoxon_p_one_sided']:.4g}. "
        f"Architecture cost {'fully' if arch_explains_diff else 'does NOT fully'} "
        f"explains the canonical-vs-diff-test gap (|Δarch − Δdiff_test| "
        f"{'≤' if arch_explains_diff else '>'} max(σ_arch, σ_diff_test))."
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "--canonical-root",
        type=Path,
        default=Path("outputs/redesign/p5_canonical_seed42"),
        help="Canonical run root (contains fold0/...fold4/best_summary.json).",
    )
    p.add_argument(
        "--sweep-root",
        type=Path,
        default=Path("outputs/redesign/p5_entropy_reg"),
        help="Entropy-reg sweep root (contains lambda_*/fold*/summary.json).",
    )
    p.add_argument(
        "--diff-test-root",
        type=Path,
        default=Path("outputs/redesign/p5_diff_test"),
        help="Diff-test run root (contains fold*/summary.json).",
    )
    p.add_argument(
        "--out-path",
        type=Path,
        default=Path(
            "outputs/redesign/interpretability/"
            "architecture_vs_regularization_decomposition.json"
        ),
    )
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )

    canonical_root: Path = args.canonical_root
    sweep_root: Path = args.sweep_root
    diff_test_root: Path = args.diff_test_root
    out_path: Path = args.out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Read inputs --------------------------------------------------------
    logger.info("Reading canonical from %s", canonical_root)
    canonical_r2 = _read_canonical_per_fold(canonical_root)
    logger.info("Canonical per-fold R²: %s", canonical_r2)

    sweep_r2_per_lambda: dict[str, list[float]] = {}
    for lambda_dir, lambda_val in _LAMBDA_DIR_TO_VALUE:
        logger.info("Reading sweep %s from %s", lambda_dir, sweep_root)
        sweep_r2_per_lambda[lambda_dir] = _read_sweep_per_fold(sweep_root, lambda_dir)
        logger.info(
            "Sweep λ=%s per-fold R²: %s", lambda_val, sweep_r2_per_lambda[lambda_dir]
        )

    logger.info("Reading diff-test from %s", diff_test_root)
    diff_test_r2 = _read_diff_test_per_fold(diff_test_root)
    logger.info("Diff-test per-fold R²: %s", diff_test_r2)

    # 2. Aggregate stats ----------------------------------------------------
    canonical_summary = _summarize(canonical_r2)
    sweep_per_lambda: dict[str, dict[str, Any]] = {}
    for lambda_dir, lambda_val in _LAMBDA_DIR_TO_VALUE:
        block = _summarize(sweep_r2_per_lambda[lambda_dir])
        block["lambda"] = lambda_val
        block["lambda_dir"] = lambda_dir
        # Use the literal numeric string as the JSON key so downstream paper
        # tooling can index without the encoded directory name.
        sweep_per_lambda[str(lambda_val)] = block
    diff_test_summary = _summarize(diff_test_r2)

    # 3. Decomposition + paired Wilcoxon ------------------------------------
    sweep_l0 = sweep_r2_per_lambda["lambda_0"]
    sweep_lmax = sweep_r2_per_lambda["lambda_1p0"]
    architecture_cost = _paired_diff_block(canonical_r2, sweep_l0)
    regularization_cost_max = _paired_diff_block(sweep_l0, sweep_lmax)
    diff_test_cost = _paired_diff_block(canonical_r2, diff_test_r2)

    # 4. Interpretation -----------------------------------------------------
    interpretation = _build_interpretation(
        canonical_mean=canonical_summary["mean"],
        canonical_std=canonical_summary["std"],
        sweep_l0_mean=sweep_per_lambda["0.0"]["mean"],
        sweep_lmax_mean=sweep_per_lambda["1.0"]["mean"],
        diff_test_mean=diff_test_summary["mean"],
        arch_cost=architecture_cost,
        reg_cost=regularization_cost_max,
        diff_test_cost=diff_test_cost,
    )

    # 5. Compose output JSON -----------------------------------------------
    out: dict[str, Any] = {
        "method": "decomposition_via_paired_per_fold_differences",
        "n_folds": _N_FOLDS,
        # Resolve the worktree root from this file's location so the SHA
        # does not depend on the user's CWD when invoking the script.
        "git_commit": git_sha(Path(__file__).resolve().parents[3]),
        "inputs": {
            "canonical_root": str(canonical_root),
            "sweep_root": str(sweep_root),
            "diff_test_root": str(diff_test_root),
        },
        "lambda_grid": [v for _, v in _LAMBDA_DIR_TO_VALUE],
        "canonical": canonical_summary,
        "sweep_per_lambda": sweep_per_lambda,
        "diff_test": diff_test_summary,
        "decomposition": {
            "architecture_cost": architecture_cost,
            "regularization_cost_max_lambda": regularization_cost_max,
            "diff_test_cost": diff_test_cost,
        },
        "interpretation": interpretation,
    }
    out_path.write_text(json.dumps(out, indent=2))
    logger.info("Wrote %s", out_path)
    logger.info("Interpretation: %s", interpretation)
    return 0


if __name__ == "__main__":
    sys.exit(main())
