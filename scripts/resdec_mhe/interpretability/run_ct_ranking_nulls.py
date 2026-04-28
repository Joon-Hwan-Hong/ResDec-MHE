"""Three CT-ranking null distributions: LOCO + Wasserstein + CMI bootstrap.

Three sub-analyses run in parallel:

1. **LOCO ranking null** — for n_perms iterations, randomly permute the
   assignment of LOCO ΔR² values across the 31 CTs, recompute the Spearman
   concordance with the canonical (well-covered) ranking, build empirical
   distribution.  Tests the null "any CT could be rank-1 by random chance."

2. **Wasserstein within-CT label-shuffle null** — per CT × per gene, shuffle
   the resilient/vulnerable labels n_perms times, recompute Wasserstein-1
   distance, build per-gene empirical null.  Top-20 lowest-p genes per CT
   exported.

3. **CMI bootstrap CI** — n_boot subject resamples (with replacement),
   recompute KSG conditional MI per CT, derive 95% CI per CT.  Calibrates
   the rank-of-Splatter claim under sampling variability.

Output:
   ``outputs/redesign/interpretability/ct_ranking_nulls/{loco_ranking_null.json, wasserstein_perm_pvalues_per_ct_top20.csv, cmi_bootstrap_ci.json, summary.json}``.
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

_WORKTREE_ROOT = Path(__file__).resolve().parents[3]
if str(_WORKTREE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKTREE_ROOT))

from src.analysis.conditional_mi import conditional_mi_per_celltype
from src.analysis.pseudobulk_io import load_pseudobulk_matrix
from src.data.constants import CELL_TYPE_ORDER
from src.utils.provenance import git_sha


def _wasserstein_per_gene(expr_a: np.ndarray, expr_b: np.ndarray) -> np.ndarray:
    """Per-gene Wasserstein-1 distance between two subject groups; NaN-safe.

    Vectorized via the closed form for 1-D Wasserstein-1 between equal-sized
    samples: ``W₁ = mean(|sort(a) - sort(b)|)``. For label-shuffle perms,
    ``len(expr_a) == len(expr_b)`` always holds, so this is exact. Falls
    back to the per-gene scipy call for the (rare) unequal-size case where
    the closed form does not apply.
    """
    n_genes = expr_a.shape[1]
    out = np.full(n_genes, np.nan)
    finite_a = np.isfinite(expr_a).sum(axis=0) >= 3
    finite_b = np.isfinite(expr_b).sum(axis=0) >= 3
    ok = finite_a & finite_b
    if not ok.any():
        return out
    a = expr_a[:, ok]
    b = expr_b[:, ok]
    if a.shape[0] == b.shape[0] and not (np.isnan(a).any() or np.isnan(b).any()):
        # Vectorized equal-sized closed form.
        a_sorted = np.sort(a, axis=0)
        b_sorted = np.sort(b, axis=0)
        out[ok] = np.abs(a_sorted - b_sorted).mean(axis=0)
        return out
    # Mixed-size or NaN-bearing fallback: per-gene scipy call only on the
    # surviving genes (still much smaller than the original 4785-gene loop).
    ok_idx = np.where(ok)[0]
    for k, j in enumerate(ok_idx):
        col_a = a[:, k][np.isfinite(a[:, k])]
        col_b = b[:, k][np.isfinite(b[:, k])]
        if len(col_a) < 3 or len(col_b) < 3:
            continue
        try:
            out[j] = float(stats.wasserstein_distance(col_a, col_b))
        except Exception:
            continue
    return out

logger = logging.getLogger(__name__)


# ─── 1. LOCO RANKING NULL ──────────────────────────────────────────────────────

def loco_ranking_null(
    loco_json_path: Path,
    n_perms: int,
    seed: int,
    well_covered_cts: set[str],
) -> dict:
    """Empirical p-value for "Splatter is rank-1 in LOCO by chance."

    Test statistic: rank of Splatter (1=most load-bearing = most negative ΔR²).
    Null hypothesis: ΔR² assignment to CTs is exchangeable across CTs.
    Empirical p = (1 + #perms with rank ≤ obs_rank) / (n_perms + 1).
    """
    rng = np.random.default_rng(seed)
    with loco_json_path.open() as f:
        d = json.load(f)
    pc = d["per_cell_type"]
    delta_r2 = np.array([e["delta_r2_vs_canonical"] for e in pc])
    cts = [e["cell_type"] for e in pc]
    keep_mask = np.array([ct in well_covered_cts for ct in cts])
    delta_well = delta_r2[keep_mask]
    cts_well = [c for c, k in zip(cts, keep_mask) if k]

    splatter_idx = cts_well.index("Splatter") if "Splatter" in cts_well else None
    if splatter_idx is None:
        return {"error": "Splatter not in well-covered set"}

    # Observed rank of Splatter (1 = most load-bearing = most negative ΔR²)
    order = np.argsort(delta_well)  # ascending; most negative first
    obs_rank = int(np.where(order == splatter_idx)[0][0]) + 1

    # Permutation: shuffle ΔR² across CTs
    perm_ranks = np.empty(n_perms, dtype=np.int64)
    for k in range(n_perms):
        permuted = rng.permutation(delta_well)
        order_p = np.argsort(permuted)
        # The Splatter "slot" stays fixed; we ask "would the value at that slot rank #1"
        # under permuted assignment of values to CTs?  Equivalent: rank of value originally
        # at splatter_idx = rank of permuted[splatter_idx] in the permuted distribution.
        # But after permutation, EVERY slot has a random value. So rank of Splatter slot
        # under permutation = rank(permuted[splatter_idx]) = uniform over 1..n.
        # That's correct for the exchangeability null.
        perm_ranks[k] = int(np.where(order_p == splatter_idx)[0][0]) + 1

    p_emp = (1 + (perm_ranks <= obs_rank).sum()) / (n_perms + 1)
    return {
        "test_ct": "Splatter",
        "n_well_covered_cts": len(cts_well),
        "obs_rank": obs_rank,
        "obs_delta_r2": float(delta_well[splatter_idx]),
        "n_perms": n_perms,
        "perm_rank_mean": float(perm_ranks.mean()),
        "perm_rank_p25_p50_p75": [float(np.quantile(perm_ranks, q)) for q in [0.25, 0.5, 0.75]],
        "p_empirical": float(p_emp),
        "interpretation": (
            "p_empirical = probability under exchangeability of ΔR² across CTs "
            "that a random CT is rank ≤ obs_rank for Splatter."
        ),
    }


# ─── 2. WASSERSTEIN WITHIN-CT LABEL-SHUFFLE NULL ────────────────────────────────

def _wasserstein_perm_one_ct(
    expr: np.ndarray,
    is_res: np.ndarray,
    n_perms: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-CT label-shuffle null on Wasserstein-1.  Returns (obs_w1, p_perm) per gene.

    Computes per-gene W1 directly via scipy.stats.wasserstein_distance — does
    NOT rely on src.analysis.resilience_distributional.wasserstein_per_celltype
    (which collapses per-gene values into mean + top-10 only).
    """
    rng = np.random.default_rng(seed)
    obs_w1 = _wasserstein_per_gene(expr[is_res], expr[~is_res])

    n_genes = expr.shape[1]
    finite = np.isfinite(obs_w1)
    ge_count = np.zeros(n_genes, dtype=np.int64)
    for k in range(n_perms):
        perm_idx = rng.permutation(len(is_res))
        perm_res = is_res[perm_idx]
        w1_perm = _wasserstein_per_gene(expr[perm_res], expr[~perm_res])
        ge_count[finite] += (w1_perm[finite] >= obs_w1[finite]).astype(np.int64)
    p_perm = (1 + ge_count) / (n_perms + 1)
    p_perm[~finite] = np.nan
    return obs_w1, p_perm


# ─── 3. CMI BOOTSTRAP CI ────────────────────────────────────────────────────────

def cmi_bootstrap_ci(
    pb: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    n_boot: int,
    seed: int,
    n_jobs: int,
) -> dict:
    """Bootstrap CI on per-CT raw-pseudobulk conditional MI.

    Returns per-CT 2.5/50/97.5 percentiles + observed value.
    """
    rng = np.random.default_rng(seed)
    n_subj = pb.shape[0]
    # Observed
    obs = conditional_mi_per_celltype(
        pb, y, z, aggregation="max", regressor="linear", n_jobs=n_jobs,
        cell_type_names=CELL_TYPE_ORDER,
    )
    obs_cmi = {e["cell_type"]: e["conditional_mi_given_pathology"] for e in obs["per_cell_type"]}

    # Bootstrap with heartbeat: log every HEARTBEAT_EVERY iters so a stuck
    # bootstrap can be diagnosed externally (otherwise the inner CMI loop is
    # silent for the entire wall, and a hung joblib worker looks identical to
    # progress).
    HEARTBEAT_EVERY = max(1, n_boot // 20)  # ~5% granularity
    import time as _time
    _t0 = _time.time()
    boot_per_ct: dict[str, list[float]] = {ct: [] for ct in obs_cmi}
    for b in range(n_boot):
        idx = rng.integers(0, n_subj, size=n_subj)
        try:
            r = conditional_mi_per_celltype(
                pb[idx], y[idx], z[idx], aggregation="max",
                regressor="linear", n_jobs=n_jobs,
                cell_type_names=CELL_TYPE_ORDER,
            )
            for e in r["per_cell_type"]:
                boot_per_ct[e["cell_type"]].append(e["conditional_mi_given_pathology"])
        except Exception as exc:
            logger.warning("boot %d failed: %s", b, exc)
            continue
        if (b + 1) % HEARTBEAT_EVERY == 0 or b == n_boot - 1:
            elapsed = _time.time() - _t0
            eta = (n_boot - b - 1) * (elapsed / max(b + 1, 1))
            logger.info(
                "  cmi_bootstrap heartbeat: %d / %d (%.0f%%) elapsed=%.1fmin ETA=%.1fmin",
                b + 1, n_boot, 100 * (b + 1) / n_boot,
                elapsed / 60.0, eta / 60.0,
            )

    out: dict[str, dict] = {}
    for ct, vals in boot_per_ct.items():
        if len(vals) < 10:
            out[ct] = {
                "observed_cmi": obs_cmi[ct],
                "n_valid_boots": len(vals),
                "ci_2_5": None, "ci_50": None, "ci_97_5": None,
            }
        else:
            arr = np.array(vals)
            out[ct] = {
                "observed_cmi": obs_cmi[ct],
                "n_valid_boots": len(vals),
                "ci_2_5": float(np.quantile(arr, 0.025)),
                "ci_50": float(np.quantile(arr, 0.50)),
                "ci_97_5": float(np.quantile(arr, 0.975)),
            }
    return {"n_boot": n_boot, "n_subjects": n_subj, "per_ct": out}


# ─── DRIVER ─────────────────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "--loco-json",
        default="outputs/redesign/interpretability/loco_zero_out/loco_per_celltype.json",
    )
    p.add_argument(
        "--coverage-json",
        default="outputs/redesign/interpretability/ct_coverage_full_cohort.json",
    )
    p.add_argument(
        "--residual-csv",
        default="outputs/redesign/interpretability/residual_per_subject.csv",
    )
    p.add_argument("--precomputed-dir", default="data/precomputed")
    p.add_argument("--gene-names-npy", default="data/precomputed/gene_names.npy")
    p.add_argument(
        "--metadata-csv",
        default="data/metadata_ROSMAP/metadata.csv",
    )
    p.add_argument(
        "--pred-root",
        default="outputs/redesign/p5_canonical_seed42",
        help="Per-fold predictions root used by cmi_bootstrap to load val_predictions_best.npz",
    )
    p.add_argument(
        "--tabpfn-dir",
        default="data/redesign",
        help="Directory containing tabpfn_outer_fold{0..4}.npz (cmi_bootstrap only)",
    )
    p.add_argument(
        "--out-dir",
        default="outputs/redesign/interpretability/ct_ranking_nulls",
    )
    p.add_argument("--n-perms", type=int, default=1000)
    p.add_argument("--n-boot", type=int, default=200)
    p.add_argument("--n-jobs", type=int, default=4)
    p.add_argument("--quartile-fraction", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--analyses", nargs="+",
        default=["loco_null", "wasserstein_null", "cmi_bootstrap"],
        choices=["loco_null", "wasserstein_null", "cmi_bootstrap"],
    )
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # Load coverage filter
    with Path(args.coverage_json).open() as f:
        cov = json.load(f)
    well_covered = {ct for ct, info in cov["per_ct"].items() if info["well_covered"]}
    logger.info("well-covered set: %d CTs", len(well_covered))

    summary: dict = {
        "n_perms": args.n_perms,
        "n_boot": args.n_boot,
        "n_jobs": args.n_jobs,
        "git_commit": git_sha(_WORKTREE_ROOT),
        "analyses_run": args.analyses,
    }

    # ─── LOCO ranking null ──────────────────────────────────────────────────
    if "loco_null" in args.analyses:
        logger.info("LOCO ranking null...")
        loco_result = loco_ranking_null(
            Path(args.loco_json), args.n_perms, args.seed, well_covered,
        )
        (out_dir / "loco_ranking_null.json").write_text(
            json.dumps(loco_result, indent=2)
        )
        logger.info("  Splatter obs_rank=%s, p_empirical=%.4f",
                    loco_result.get("obs_rank"),
                    loco_result.get("p_empirical", float("nan")))
        summary["loco_ranking_null"] = loco_result

    # ─── Wasserstein within-CT label-shuffle ─────────────────────────────────
    if "wasserstein_null" in args.analyses:
        logger.info("Wasserstein within-CT label-shuffle null...")
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
        pb = load_pseudobulk_matrix(Path(args.precomputed_dir), keep_ids)
        gene_names = list(np.load(args.gene_names_npy, allow_pickle=True))

        def _ct_job(ct: int):
            return _wasserstein_perm_one_ct(
                pb[:, ct, :], is_res, args.n_perms, args.seed + ct,
            )

        results = Parallel(n_jobs=args.n_jobs, verbose=10)(
            delayed(_ct_job)(ct) for ct in range(pb.shape[1])
        )
        perm_rows = []
        skipped_per_ct: dict[str, int] = {}
        for ct, (obs_w1, p_perm) in enumerate(results):
            ct_name = CELL_TYPE_ORDER[ct] if ct < len(CELL_TYPE_ORDER) else f"CT_{ct}"
            skipped_per_ct[ct_name] = int((~np.isfinite(p_perm)).sum())
            for j in range(len(p_perm)):
                if not np.isfinite(p_perm[j]):
                    continue
                perm_rows.append({
                    "cell_type_index": ct,
                    "cell_type": ct_name,
                    "gene": gene_names[j] if j < len(gene_names) else f"gene_{j}",
                    "wasserstein_observed": float(obs_w1[j]),
                    "p_perm": float(p_perm[j]),
                })
        summary["wasserstein_skipped_genes_per_ct"] = skipped_per_ct
        perm_df = pd.DataFrame(perm_rows)
        if len(perm_df):
            perm_df = perm_df.sort_values(["cell_type_index", "p_perm"]).groupby(
                "cell_type_index", as_index=False, group_keys=False,
            ).head(20)
        perm_df.to_csv(out_dir / "wasserstein_perm_pvalues_per_ct_top20.csv", index=False)
        logger.info("  wrote %s (%d rows)",
                    out_dir / "wasserstein_perm_pvalues_per_ct_top20.csv",
                    len(perm_df))
        summary["wasserstein_null_csv"] = str(
            out_dir / "wasserstein_perm_pvalues_per_ct_top20.csv"
        )

    # ─── CMI bootstrap CI ────────────────────────────────────────────────────
    if "cmi_bootstrap" in args.analyses:
        logger.info("CMI bootstrap CI (n_boot=%d)...", args.n_boot)
        # Load full N=516 pseudobulk + composite Y + pathology Z
        # Subjects: union of all val folds
        files = sorted(Path(args.precomputed_dir).glob("R*.pt"))
        all_ids = [f.stem for f in files]
        pb_full = load_pseudobulk_matrix(Path(args.precomputed_dir), all_ids)

        # Load composite Y: from val_predictions_best.npz across all 5 folds.
        # pred_root + tabpfn_dir are env-driven via argparse to avoid hardcoded
        # paths (memory rule feedback_no_hardcoded_paths.md).
        pred_root = Path(args.pred_root)
        tabpfn_dir = Path(args.tabpfn_dir)
        composite_y = {}
        for fold in range(5):
            v = np.load(pred_root / f"fold{fold}/val_predictions_best.npz", allow_pickle=True)
            t = np.load(tabpfn_dir / f"tabpfn_outer_fold{fold}.npz", allow_pickle=True)
            for sid, p, tp in zip(v["subject_ids"], v["predictions"], t["y_tabpfn"]):
                composite_y[str(sid)] = float(p + tp)
        y = np.array([composite_y.get(sid, np.nan) for sid in all_ids], dtype=np.float64)

        # Pathology Z from metadata
        meta = pd.read_csv(args.metadata_csv)
        # Subject-ID column resolution: prefer canonical name, fall back through
        # an explicit list, raise rather than silently use index column.
        for cand in ("ROSMAP_IndividualID", "subject_id", "subject"):
            if cand in meta.columns:
                id_col_meta = cand
                break
        else:
            raise ValueError(
                f"metadata CSV missing subject-ID column; expected one of "
                f"['ROSMAP_IndividualID', 'subject_id', 'subject'], got {list(meta.columns)}"
            )
        meta = meta.set_index(id_col_meta)
        z_cols = ["gpath", "amylsqrt", "tangsqrt"]
        z = np.full((len(all_ids), len(z_cols)), np.nan)
        for i, sid in enumerate(all_ids):
            if sid in meta.index:
                row = meta.loc[sid]
                for j, c in enumerate(z_cols):
                    if c in meta.columns:
                        z[i, j] = float(row[c]) if pd.notna(row[c]) else np.nan

        # Drop subjects with missing Y or Z
        keep_mask = np.isfinite(y) & np.all(np.isfinite(z), axis=1)
        pb_keep = pb_full[keep_mask]
        y_keep = y[keep_mask]
        z_keep = z[keep_mask]
        logger.info("  CMI bootstrap on n=%d subjects with valid Y+Z", keep_mask.sum())

        boot = cmi_bootstrap_ci(pb_keep, y_keep, z_keep, args.n_boot, args.seed, args.n_jobs)
        (out_dir / "cmi_bootstrap_ci.json").write_text(json.dumps(boot, indent=2))
        logger.info("  wrote %s (%d CTs)",
                    out_dir / "cmi_bootstrap_ci.json",
                    len(boot.get("per_ct", {})))
        summary["cmi_bootstrap_n_subjects"] = int(keep_mask.sum())

    summary["elapsed_min"] = round((time.time() - t0) / 60, 2)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("done in %.1f min — see %s", (time.time() - t0) / 60, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
