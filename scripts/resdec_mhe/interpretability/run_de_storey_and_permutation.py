"""Storey q-values + per-CT permutation null on resilient-vs-vulnerable DE.

Two analyses on the same data:

1. **Storey q-values** — alternative to BH-FDR that estimates π₀ (fraction of
   true nulls) via the smoother method (Storey & Tibshirani 2003).  More
   powerful than BH when π₀ < 1.  Per-CT (within-CT correction across the
   4,785 genes), since DE is run per CT.

2. **Per-CT permutation null on Wilcoxon stat** — for each CT, shuffle the
   resilient/vulnerable subject labels ``n_perms`` times, recompute the
   Wilcoxon test statistic per gene, build an empirical null distribution.
   Empirical p_perm = (1 + #|stat_perm| ≥ |stat_obs|) / (n_perms + 1).
   Robust to non-normality + dependence.

Both run in parallel across CTs via joblib.

Output: ``outputs/canonical/interpretability/de_storey_and_permutation/{storey_qvalues_per_ct.csv, perm_pvalues_per_ct.csv, summary.json}``.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from scipy import stats
from scipy.interpolate import UnivariateSpline

_WORKTREE_ROOT = Path(__file__).resolve().parents[3]
if str(_WORKTREE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKTREE_ROOT))

from src.analysis.pseudobulk_io import load_pseudobulk_matrix
from src.data.constants import CELL_TYPE_ORDER
from src.utils.provenance import git_sha

logger = logging.getLogger(__name__)


def storey_qvalues(p: np.ndarray, lambdas: np.ndarray = None) -> tuple[np.ndarray, float, str]:
    """Storey-Tibshirani 2003 q-values via the smoother method (default in R qvalue).

    Returns ``(q_values, pi0_estimate, method)``. ``method`` is one of:
      * ``"storey_smoother"`` — full Storey procedure with the cubic-spline
        π₀ estimate.
      * ``"storey_smoother_mean_fallback"`` — spline fit failed; π₀ is the
        mean of the per-λ estimates (less principled — see WARNING in log).
      * ``"bh_fallback_few_pvalues"`` — fewer than 10 finite p-values; fell
        back to BH (π₀=1) which is equivalent to Storey at π₀=1.

    Implementation: estimate π₀(λ) = #{p_i > λ} / (m·(1−λ)) on a grid of λ values,
    fit a natural cubic spline, and evaluate at λ → 1 (use λ = 0.95 as default).
    """
    p = np.asarray(p, dtype=np.float64)
    finite = np.isfinite(p)
    if finite.sum() < 10:
        # not enough data for smoother — fall back to π₀=1 (BH equivalent)
        q = np.full_like(p, np.nan)
        if finite.any():
            ranked = stats.false_discovery_control(p[finite], method="bh")
            q[finite] = ranked
        return q, 1.0, "bh_fallback_few_pvalues"
    p_valid = p[finite]
    m = len(p_valid)

    if lambdas is None:
        lambdas = np.arange(0.05, 0.96, 0.05)
    pi0_lam = np.array([(p_valid > lam).sum() / (m * (1.0 - lam)) for lam in lambdas])
    pi0_lam = np.clip(pi0_lam, 0.0, 1.0)
    # Smoother: cubic spline fit, evaluate at λ=max(λ_grid). On spline failure
    # we fall back to mean(pi0_lam) and emit a WARNING (this is a quiet method
    # substitution from R's qvalue::pi0est, which uses bootstrap-based selection
    # if the spline fails — see R qvalue source). Log so reviewers can spot.
    try:
        spline = UnivariateSpline(lambdas, pi0_lam, k=3, s=0)
        pi0 = float(spline(lambdas.max()))
        method = "storey_smoother"
    except Exception as e:
        logger.warning(
            "Storey spline fit failed (%s); falling back to mean(pi0_lam) = %.3f. "
            "This is a less-principled substitution; consider re-running with a "
            "smaller lambda grid or a bootstrap-based π₀ estimator.",
            e, float(pi0_lam.mean()),
        )
        pi0 = float(pi0_lam.mean())
        method = "storey_smoother_mean_fallback"
    pi0 = float(np.clip(pi0, 1e-3, 1.0))

    # Order p-values, compute q
    order = np.argsort(p_valid)
    q_ordered = np.empty(m)
    p_ordered = p_valid[order]
    # Storey q-value formula (smoothed pi0): q_(i) = pi0 * m * p_(i) / i (with monotonic adjustment)
    raw = pi0 * m * p_ordered / np.arange(1, m + 1)
    # Enforce monotonicity (q_(i) should be non-decreasing as p increases)
    q_ordered = np.minimum.accumulate(raw[::-1])[::-1]
    q_ordered = np.clip(q_ordered, 0.0, 1.0)
    q_full_valid = np.empty(m)
    q_full_valid[order] = q_ordered
    q = np.full_like(p, np.nan)
    q[finite] = q_full_valid
    return q, pi0, method


def _wilcoxon_per_gene(expr_res: np.ndarray, expr_vul: np.ndarray) -> np.ndarray:
    """Two-sample Wilcoxon rank-sum test per gene; returns U statistic per gene.

    Vectorized over genes via ``scipy.stats.mannwhitneyu(..., axis=0)`` to
    avoid the Python-loop overhead (4,785 genes × 1,000 perms = 4.78 M scipy
    calls otherwise). The NaN pattern is column-static (not perm-dependent),
    so we pre-filter genes once before delegating to the C-level vectorized
    impl. Numerically identical to the per-gene version on surviving columns.
    """
    n_genes = expr_res.shape[1]
    out = np.full(n_genes, np.nan)
    finite_res = np.isfinite(expr_res).sum(axis=0) >= 3
    finite_vul = np.isfinite(expr_vul).sum(axis=0) >= 3
    ok = finite_res & finite_vul
    if not ok.any():
        return out
    try:
        stat = stats.mannwhitneyu(
            expr_res[:, ok], expr_vul[:, ok],
            alternative="two-sided", axis=0, nan_policy="omit",
        ).statistic
        out[ok] = stat
    except Exception:
        pass
    return out


def _perm_null_one_ct(
    expr: np.ndarray,
    is_resilient: np.ndarray,
    n_perms: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-CT permutation null. Returns (observed_stats, perm_pvalues) per gene.

    Empirical p = (1 + #|stat_perm| ≥ |stat_obs - centre|) / (n_perms + 1).
    Two-sided based on Mann-Whitney U distance from the no-effect centre m·n/2.
    """
    rng = np.random.default_rng(seed)
    expr = np.asarray(expr, dtype=np.float32)
    is_res = np.asarray(is_resilient, dtype=bool)
    n_res = is_res.sum()
    n_vul = (~is_res).sum()
    if n_res < 3 or n_vul < 3:
        return np.full(expr.shape[1], np.nan), np.full(expr.shape[1], np.nan)
    # Observed
    obs = _wilcoxon_per_gene(expr[is_res], expr[~is_res])
    centre = n_res * n_vul / 2.0
    abs_obs = np.abs(obs - centre)

    # Perm null
    n_genes = expr.shape[1]
    ge_count = np.zeros(n_genes, dtype=np.int64)
    finite = np.isfinite(obs)
    for k in range(n_perms):
        perm_idx = rng.permutation(len(is_res))
        perm_res = is_res[perm_idx]
        s = _wilcoxon_per_gene(expr[perm_res], expr[~perm_res])
        ge_count[finite] += (np.abs(s[finite] - centre) >= abs_obs[finite]).astype(np.int64)
    p_perm = (1 + ge_count) / (n_perms + 1)
    p_perm[~finite] = np.nan
    return obs, p_perm


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "--residual-csv", type=Path,
        default=(
            _WORKTREE_ROOT
            / "outputs/canonical/interpretability/residual_per_subject.csv"
        ),
    )
    p.add_argument(
        "--precomputed-dir", type=Path,
        default=_WORKTREE_ROOT / "data/precomputed",
    )
    p.add_argument(
        "--gene-names-npy", type=Path,
        default=_WORKTREE_ROOT / "data/precomputed/gene_names.npy",
    )
    p.add_argument(
        "--de-input-dir", type=Path,
        default=(
            _WORKTREE_ROOT
            / "outputs/canonical/interpretability/de_resilient_vs_vulnerable"
        ),
        help="Existing per-CT Wilcoxon DE outputs (reads CT_*_de.csv for p-values).",
    )
    p.add_argument(
        "--out-dir", type=Path,
        default=(
            _WORKTREE_ROOT
            / "outputs/canonical/interpretability/de_storey_and_permutation"
        ),
    )
    p.add_argument("--quartile-fraction", type=float, default=0.25)
    p.add_argument("--n-perms", type=int, default=1000)
    p.add_argument("--n-jobs", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # ─── 1. STOREY Q-VALUES ON EXISTING WILCOXON P-VALUES ────────────────────
    logger.info("Storey q-values per CT...")
    storey_rows = []
    pi0_per_ct: dict[str, float] = {}
    qvalue_method_per_ct: dict[str, str] = {}
    for ct_idx in range(len(CELL_TYPE_ORDER)):
        de_csv = Path(args.de_input_dir) / f"CT_{ct_idx:02d}_de.csv"
        if not de_csv.exists():
            continue
        df = pd.read_csv(de_csv)
        p_vals = df["p_value"].to_numpy()
        q, pi0, qmethod = storey_qvalues(p_vals)
        df["q_storey"] = q
        df["cell_type_index"] = ct_idx
        df["cell_type"] = CELL_TYPE_ORDER[ct_idx]
        df["pi0_estimate"] = pi0
        df["qvalue_method"] = qmethod  # storey_smoother | …_mean_fallback | bh_fallback…
        pi0_per_ct[CELL_TYPE_ORDER[ct_idx]] = pi0
        qvalue_method_per_ct[CELL_TYPE_ORDER[ct_idx]] = qmethod
        # Top-20 by q
        top = df.nsmallest(20, "q_storey")
        storey_rows.append(top)
    storey_df = pd.concat(storey_rows, ignore_index=True) if storey_rows else pd.DataFrame()
    storey_df.to_csv(out_dir / "storey_qvalues_per_ct_top20.csv", index=False)
    logger.info(
        "  wrote %s (%d rows, mean π₀=%.3f)",
        out_dir / "storey_qvalues_per_ct_top20.csv",
        len(storey_df), float(np.mean(list(pi0_per_ct.values()))) if pi0_per_ct else float("nan"),
    )

    # ─── 2. PER-CT PERMUTATION NULL ON WILCOXON STAT ─────────────────────────
    logger.info("Permutation null (n_perms=%d) per CT...", args.n_perms)
    res_df = pd.read_csv(args.residual_csv)
    id_col = "ROSMAP_IndividualID" if "ROSMAP_IndividualID" in res_df.columns else res_df.columns[0]
    res_df = res_df.rename(columns={id_col: "subject_id"})
    finite = np.isfinite(res_df["residual"])
    q_lo = res_df.loc[finite, "residual"].quantile(args.quartile_fraction)
    q_hi = res_df.loc[finite, "residual"].quantile(1 - args.quartile_fraction)
    res_df["group"] = "middle"
    res_df.loc[res_df["residual"] >= q_hi, "group"] = "resilient"
    res_df.loc[res_df["residual"] <= q_lo, "group"] = "vulnerable"
    keep = res_df[res_df["group"].isin(("resilient", "vulnerable"))].copy()
    keep_ids = keep["subject_id"].astype(str).tolist()
    is_res = (keep["group"] == "resilient").to_numpy()
    pb = load_pseudobulk_matrix(Path(args.precomputed_dir), keep_ids)  # (n_subj, n_ct, n_gene)
    n_ct = pb.shape[1]
    gene_names = list(np.load(args.gene_names_npy, allow_pickle=True))

    def _ct_job(ct: int):
        return _perm_null_one_ct(pb[:, ct, :], is_res, args.n_perms, args.seed + ct)

    results = Parallel(n_jobs=args.n_jobs, verbose=10)(
        delayed(_ct_job)(ct) for ct in range(n_ct)
    )

    perm_rows = []
    for ct, (obs, p_perm) in enumerate(results):
        for j in range(len(p_perm)):
            if not np.isfinite(p_perm[j]):
                continue
            perm_rows.append({
                "cell_type_index": ct,
                "cell_type": CELL_TYPE_ORDER[ct] if ct < len(CELL_TYPE_ORDER) else f"CT_{ct}",
                "gene": gene_names[j] if j < len(gene_names) else f"gene_{j}",
                "wilcoxon_U_observed": float(obs[j]),
                "p_perm": float(p_perm[j]),
            })
    perm_df = pd.DataFrame(perm_rows)
    # Keep only top-50 lowest p_perm per CT to keep CSV small
    if len(perm_df):
        perm_df = perm_df.sort_values(["cell_type_index", "p_perm"]).groupby(
            "cell_type_index", as_index=False, group_keys=False
        ).head(50)
    perm_df.to_csv(out_dir / "perm_pvalues_per_ct_top50.csv", index=False)
    logger.info("  wrote %s (%d rows)", out_dir / "perm_pvalues_per_ct_top50.csv", len(perm_df))

    # ─── 3. SUMMARY ──────────────────────────────────────────────────────────
    summary = {
        "n_perms": args.n_perms,
        "n_jobs": args.n_jobs,
        "quartile_fraction": args.quartile_fraction,
        "n_resilient": int(is_res.sum()),
        "n_vulnerable": int((~is_res).sum()),
        "n_cts": int(n_ct),
        "n_genes": pb.shape[2],
        "pi0_per_ct": pi0_per_ct,
        "mean_pi0": float(np.mean(list(pi0_per_ct.values()))) if pi0_per_ct else None,
        # Per-CT Storey-vs-BH-fallback method (so consumers know which q-values
        # are bona-fide Storey vs the BH fallback that fires when n_finite < 10
        # OR the spline fit fails).
        "qvalue_method_per_ct": qvalue_method_per_ct,
        "storey_qvalues_csv": str(out_dir / "storey_qvalues_per_ct_top20.csv"),
        "perm_pvalues_csv": str(out_dir / "perm_pvalues_per_ct_top50.csv"),
        "git_commit": git_sha(_WORKTREE_ROOT),
        "elapsed_min": round((time.time() - t0) / 60, 2),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("done in %.1f min", (time.time() - t0) / 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
