"""Subsample-without-replacement bootstrap for KSG conditional MI per cell type.

Background
----------
The KSG-MI estimator under standard bootstrap-with-replacement is UPWARD-BIASED
because duplicate points (the same subject sampled twice) have pairwise
distance zero, which inflates the k-NN counts that drive the digamma sum.
Empirically in our run that bias is ~+0.28-0.30 nats uniform across CTs, and
it is large enough that the basic-bootstrap reflection (``2·obs - q``)
over-corrects to impossible negative-CMI bounds.

Subsampling-without-replacement (Politis & Romano 1994; Politis, Romano &
Wolf 1999) avoids the duplicate-point artifact entirely: each bootstrap
draw of size ``m < n`` contains only distinct subjects, so the KSG digamma
counts are unbiased on each draw.

CI construction (Politis-Romano 1999, Theorem 2.1)
--------------------------------------------------
For an estimator T_n consistent at rate ``√n`` with nondegenerate limiting
distribution, the subsampled distribution at size m, recentered at the
full-sample point estimate T_n and rescaled by ``√(m/n)``, approximates the
sampling distribution of T_n at scale ``√n``. Concretely:

    P( √n · (T_n − μ) ≤ x )  ≈  P( √m · (T_m − T_n) ≤ x )

so the basic-bootstrap-style CI for T_n at level (1−α) is

    [ T_n − √(m/n) · q_{1−α/2}(T_m − T_n),
      T_n − √(m/n) · q_{α/2}  (T_m − T_n) ]

This is the canonical subsampling CI; it is bias-aware (the recentering at
T_n cancels additive bias of the bootstrap distribution, and the
``√(m/n)`` rescaling restores the correct standard error).

We also report the simpler symmetric CI

    T_n ± z_{0.975} · √(m/n) · sd(T_m)

as a robustness comparator.

Usage
-----
    PYTHONPATH=. uv run python scripts/resdec_mhe/interpretability/run_cmi_subsample_bootstrap.py \\
        --n-subsample 400 \\
        --n-boot 200 \\
        --n-jobs 32

Defaults match ``run_ct_ranking_nulls.py`` (precomputed-dir, gene-names,
metadata-csv, pred-root, tabpfn-dir).

Writes
------
``outputs/redesign/interpretability/ct_ranking_nulls/cmi_subsample_bootstrap_ci.json``
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

_WORKTREE_ROOT = Path(__file__).resolve().parents[3]
if str(_WORKTREE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKTREE_ROOT))

from src.analysis.composite_y import load_composite_y_with_sanity_check
from src.analysis.conditional_mi import conditional_mi_per_celltype
from src.analysis.pseudobulk_io import load_pseudobulk_matrix
from src.data.constants import CELL_TYPE_ORDER
from src.utils.provenance import git_sha

logger = logging.getLogger(__name__)

_Z_975 = 1.959963984540054


def cmi_subsample_bootstrap(
    pb: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    n_subsample: int,
    n_boot: int,
    seed: int,
    n_jobs: int,
) -> dict:
    """Subsample-without-replacement bootstrap for per-CT KSG conditional MI.

    Parameters
    ----------
    pb
        ``[n_subjects, n_celltypes, n_genes]`` raw pseudobulk array.
    y
        ``[n_subjects]`` composite cognition (``ŷ_tabpfn + residual``).
    z
        ``[n_subjects, n_pathology]`` conditioning variables.
    n_subsample
        Subsample size ``m`` (must be ``< n_subjects``). Choose so that
        ``m / n_subjects`` is well below 1 (we use ~0.78 by default; the
        Politis-Romano scaling holds in the limit ``m/n → 0`` but performs
        well for m/n up to ~0.8 with KSG estimators in our experience).
    n_boot
        Number of subsample draws.
    seed
        RNG seed for reproducibility.
    n_jobs
        ``conditional_mi_per_celltype`` joblib parallelism.

    Returns
    -------
    dict with keys
        ``n_boot``, ``n_subjects``, ``n_subsample``, ``per_ct``, where
        ``per_ct[ct]`` contains observed CMI, valid-boot count, raw
        subsample quantiles, the Politis-Romano CI, the symmetric SE-based CI,
        and the bias estimate.
    """
    rng = np.random.default_rng(seed)
    n_subj = pb.shape[0]
    if not (1 <= n_subsample < n_subj):
        raise ValueError(
            f"n_subsample must satisfy 1 <= m < n; got m={n_subsample}, n={n_subj}"
        )

    # Observed (KSG on full sample — unbiased point estimate).
    obs = conditional_mi_per_celltype(
        pb, y, z, aggregation="max", regressor="linear", n_jobs=n_jobs,
        cell_type_names=CELL_TYPE_ORDER,
    )
    obs_cmi = {
        e["cell_type"]: e["conditional_mi_given_pathology"]
        for e in obs["per_cell_type"]
    }

    # Subsample loop with heartbeat (every ~5%).
    HEARTBEAT_EVERY = max(1, n_boot // 20)
    t0 = time.time()
    boot_per_ct: dict[str, list[float]] = {ct: [] for ct in obs_cmi}
    for b in range(n_boot):
        idx = rng.choice(n_subj, size=n_subsample, replace=False)
        try:
            r = conditional_mi_per_celltype(
                pb[idx], y[idx], z[idx], aggregation="max",
                regressor="linear", n_jobs=n_jobs,
                cell_type_names=CELL_TYPE_ORDER,
            )
            for e in r["per_cell_type"]:
                boot_per_ct[e["cell_type"]].append(
                    e["conditional_mi_given_pathology"]
                )
        except Exception as exc:
            logger.warning("subsample %d failed: %s", b, exc)
            continue
        if (b + 1) % HEARTBEAT_EVERY == 0 or b == n_boot - 1:
            elapsed = time.time() - t0
            eta = (n_boot - b - 1) * (elapsed / max(b + 1, 1))
            logger.info(
                "  cmi_subsample_bootstrap heartbeat: %d / %d (%.0f%%) "
                "elapsed=%.1fmin ETA=%.1fmin",
                b + 1, n_boot, 100 * (b + 1) / n_boot,
                elapsed / 60.0, eta / 60.0,
            )

    # Politis-Romano scaling factor.
    scale = np.sqrt(n_subsample / n_subj)
    out: dict[str, dict] = {}
    for ct, vals in boot_per_ct.items():
        if len(vals) < 10:
            out[ct] = {
                "observed_cmi": obs_cmi[ct],
                "n_valid_boots": len(vals),
                "subsample_size": n_subsample,
                "scale_m_over_n": float(scale),
                "ci_pr_lo": None,
                "ci_pr_hi": None,
                "ci_se_lo": None,
                "ci_se_hi": None,
                "subsample_mean": None,
                "subsample_std": None,
            }
            continue
        arr = np.array(vals, dtype=np.float64)
        obs_val = obs_cmi[ct]
        centered = arr - obs_val  # T_m - T_n
        # Politis-Romano CI: invert quantiles of the recentered scaled distribution.
        q_lo = np.quantile(centered, 0.025)
        q_hi = np.quantile(centered, 0.975)
        ci_pr_lo = obs_val - scale * q_hi
        ci_pr_hi = obs_val - scale * q_lo
        # Symmetric SE-based comparator.
        sd_m = float(arr.std(ddof=1))
        se_n = scale * sd_m
        ci_se_lo = obs_val - _Z_975 * se_n
        ci_se_hi = obs_val + _Z_975 * se_n
        # Bias estimate from the subsample mean (after recentering: should be
        # ~0 if no systematic bias remains; nonzero indicates residual bias).
        subsample_mean = float(arr.mean())
        out[ct] = {
            "observed_cmi": float(obs_val),
            "n_valid_boots": len(vals),
            "subsample_size": n_subsample,
            "scale_m_over_n": float(scale),
            "ci_pr_lo": float(ci_pr_lo),
            "ci_pr_hi": float(ci_pr_hi),
            "ci_se_lo": float(ci_se_lo),
            "ci_se_hi": float(ci_se_hi),
            "subsample_mean": subsample_mean,
            "subsample_std": sd_m,
            "subsample_centered_q_2_5": float(q_lo),
            "subsample_centered_q_97_5": float(q_hi),
            "residual_bias_estimate": subsample_mean - float(obs_val),
        }
    return {
        "n_boot": n_boot,
        "n_subjects": n_subj,
        "n_subsample": n_subsample,
        "scale_m_over_n": float(scale),
        "method": "politis_romano_subsampling",
        "note": (
            "KSG-MI subsample bootstrap (without replacement). "
            "Avoids the duplicate-point upward bias of the with-replacement "
            "bootstrap. Two CIs reported: Politis-Romano basic-pivot "
            "(ci_pr_*) and a symmetric SE-based comparator (ci_se_*) "
            "scaled by sqrt(m/n)."
        ),
        "per_ct": out,
    }


def _resolve_id_col(df: pd.DataFrame) -> str:
    for cand in ("ROSMAP_IndividualID", "subject_id", "subject"):
        if cand in df.columns:
            return cand
    raise ValueError(
        f"DataFrame missing subject-ID column; expected one of "
        f"['ROSMAP_IndividualID', 'subject_id', 'subject'], got {list(df.columns)}"
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "--precomputed-dir",
        default="data/precomputed",
        help="Directory containing per-subject R*.pt pseudobulk tensors.",
    )
    p.add_argument(
        "--gene-names-npy",
        default="data/precomputed/gene_names.npy",
    )
    p.add_argument(
        "--metadata-csv",
        default="data/metadata_ROSMAP/metadata.csv",
    )
    p.add_argument(
        "--pred-root",
        default="outputs/redesign/p5_canonical_seed42",
        help="Per-fold predictions root for val_predictions_best.npz.",
    )
    p.add_argument(
        "--tabpfn-dir",
        default="data/redesign",
        help="Directory containing tabpfn_outer_fold{0..4}.npz.",
    )
    p.add_argument(
        "--out-path",
        default="outputs/redesign/interpretability/ct_ranking_nulls/cmi_subsample_bootstrap_ci.json",
    )
    p.add_argument("--n-subsample", type=int, default=400)
    p.add_argument("--n-boot", type=int, default=200)
    p.add_argument("--n-jobs", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    logger.info(
        "subsample bootstrap: m=%d / n_obs (target)=%d, n_boot=%d, n_jobs=%d",
        args.n_subsample, 516, args.n_boot, args.n_jobs,
    )

    # Load pseudobulk for all subjects on disk.
    files = sorted(Path(args.precomputed_dir).glob("R*.pt"))
    all_ids = [f.stem for f in files]
    pb_full = load_pseudobulk_matrix(Path(args.precomputed_dir), all_ids)

    # Composite Y via shared helper (runs heuristic mean/std guard AND the
    # stronger Pearson-correlation guard against the metadata target column).
    y = load_composite_y_with_sanity_check(
        pred_root=Path(args.pred_root),
        all_ids=all_ids,
        metadata_path=Path(args.metadata_csv),
    )

    # Pathology Z from metadata.
    meta = pd.read_csv(args.metadata_csv)
    id_col = _resolve_id_col(meta)
    meta = meta.set_index(id_col)
    z_cols = ["gpath", "amylsqrt", "tangsqrt"]
    z = np.full((len(all_ids), len(z_cols)), np.nan)
    for i, sid in enumerate(all_ids):
        if sid in meta.index:
            row = meta.loc[sid]
            for j, c in enumerate(z_cols):
                if c in meta.columns:
                    z[i, j] = float(row[c]) if pd.notna(row[c]) else np.nan

    # Drop subjects with missing Y or Z.
    keep_mask = np.isfinite(y) & np.all(np.isfinite(z), axis=1)
    pb_keep = pb_full[keep_mask]
    y_keep = y[keep_mask]
    z_keep = z[keep_mask]
    n_keep = int(keep_mask.sum())
    logger.info("subsample bootstrap on n=%d subjects with valid Y+Z", n_keep)

    if args.n_subsample >= n_keep:
        raise ValueError(
            f"--n-subsample={args.n_subsample} must be < n_keep={n_keep}; "
            f"choose a smaller subsample size."
        )

    boot = cmi_subsample_bootstrap(
        pb_keep, y_keep, z_keep,
        n_subsample=args.n_subsample,
        n_boot=args.n_boot,
        seed=args.seed,
        n_jobs=args.n_jobs,
    )
    boot["git_commit"] = git_sha(_WORKTREE_ROOT)
    boot["elapsed_min"] = round((time.time() - t0) / 60, 2)
    out_path.write_text(json.dumps(boot, indent=2))
    logger.info(
        "wrote %s (%d CTs, %.1f min)",
        out_path, len(boot.get("per_ct", {})),
        boot["elapsed_min"],
    )

    # Print top 10 by observed CMI.
    items = sorted(
        boot["per_ct"].items(), key=lambda kv: -kv[1]["observed_cmi"],
    )
    print("\nTop 10 CTs by observed CMI (bias-free Politis-Romano subsample CI):")
    for ct, v in items[:10]:
        if v.get("ci_pr_lo") is None:
            print(f"  {ct:42s}  obs={v['observed_cmi']:.4f}  CI=N/A (n_valid={v['n_valid_boots']})")
            continue
        print(
            f"  {ct:42s}  obs={v['observed_cmi']:.4f}  "
            f"PR-CI=[{v['ci_pr_lo']:.4f}, {v['ci_pr_hi']:.4f}]  "
            f"SE-CI=[{v['ci_se_lo']:.4f}, {v['ci_se_hi']:.4f}]"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
