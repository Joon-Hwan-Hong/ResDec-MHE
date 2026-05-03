"""Compare trained-encoder SAE vs random-encoder SAE (Heap et al. 2026 null).

Per the SAE design doc §8.3 (``docs/plans/2026-04-28-sparse-autoencoder-design.md``)
+ Orlov §4.2: for each matched config (architecture / layer / expansion / k),
report side-by-side:

* ``interpretable_candidate_fraction`` — fraction of features flagged as
  ``interpretable_candidate`` by ``interpret_features`` (Mann-Whitney
  cognition p<0.05 + non-dead + non-ubiquitous + (for fused only) one-CT
  dominant).
* ``mw_p_cognition_lt_0.05_count`` — count of features whose
  ``mw_p_cognition < 0.05`` (the cognition-only proxy, *before* the
  "interpretable" composite gate). Useful as a coarser secondary metric.
* ``dead_fraction`` — fraction of features with the ``"dead"`` flag.
* ``decoder_cos_sim_top10_pairs`` — between the two SAEs (trained vs random),
  for the top-10 features ranked by ``fraction_active`` in EACH, compute the
  pairwise cosine similarity matrix of decoder columns
  (``W_dec[:, j_trained]`` vs ``W_dec[:, j_random]``). We report:

    1. ``mean_abs_cosine_off_diag`` — the mean of |cosine| over the
       off-diagonal entries of the 10×10 matrix. Coarse "have the SAEs
       found similar concepts in arbitrary order" stat.
    2. ``hungarian_mean_diagonal_cosine`` — the canonical metric. We
       align trained-feature j to its single best random match via the
       Hungarian algorithm (``scipy.optimize.linear_sum_assignment``),
       then average the post-alignment diagonal cosine. This is the
       expected similarity AFTER permutation invariance is removed, and
       is what the Orlov §8.3 / Heap 2026 prescription calls for.

  Interpretation: LOW Hungarian-aligned mean diagonal cosine → trained and
  random SAEs find different decoder directions → trained encoder is
  contributing concept structure beyond data statistics. GOOD.

Acceptance criterion (design doc §9.2 derived from Orlov §8.3):

    trained_interpretable_fraction > 1.5 × random_interpretable_fraction

If the criterion FAILS, we must conclude (per Heap et al. 2026 / Orlov §4.2)
that the SAE is just learning data statistics, not the trained encoder's
concepts. The `pass_1.5x_criterion` field of the output JSON encodes this.

Usage
-----
    PYTHONPATH=<worktree-root> \\
    uv run python scripts/resdec_mhe/interpretability/compare_trained_vs_random_sae.py \\
        --trained-dir outputs/canonical/sae/batch_topk/fused/exp32_k64_seed0 \\
        --random-dir outputs/canonical/sae/random_encoder/batch_topk/fused/exp32_k64_seed0 \\
        --out outputs/canonical/sae/random_encoder_null/comparison.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

_WORKTREE_ROOT = Path(__file__).resolve().parents[3]
if not (_WORKTREE_ROOT / "src").is_dir():
    raise RuntimeError(
        f"sys.path bootstrap failed: {_WORKTREE_ROOT}/src not found; "
        "set PYTHONPATH=<worktree-root>."
    )
sys.path.insert(0, str(_WORKTREE_ROOT))

logger = logging.getLogger(__name__)


def _load_run(run_dir: Path) -> dict:
    """Load reconstruction_metrics.json + feature_report.json + sae_model.npz from a run dir."""
    metrics_path = run_dir / "reconstruction_metrics.json"
    report_path = run_dir / "feature_report.json"
    model_path = run_dir / "sae_model.npz"
    for f in (metrics_path, report_path, model_path):
        if not f.exists():
            raise FileNotFoundError(f"Missing {f}")

    metrics = json.loads(metrics_path.read_text())
    reports = json.loads(report_path.read_text())
    npz = np.load(model_path, allow_pickle=True)
    W_dec = np.asarray(npz["W_dec"])
    fraction_active = np.asarray(npz["stat_fraction_active"])
    return {
        "metrics": metrics,
        "reports": reports,
        "W_dec": W_dec,
        "fraction_active": fraction_active,
    }


def _interpretable_fraction(reports: list[dict]) -> float:
    if not reports:
        return float("nan")
    n = len(reports)
    n_interp = sum(1 for r in reports if "interpretable_candidate" in r.get("flags", []))
    return n_interp / n if n > 0 else float("nan")


def _mw_p_cog_lt_05_count(reports: list[dict]) -> int:
    return sum(
        1
        for r in reports
        if r.get("mw_p_cognition") is not None
        and float(r["mw_p_cognition"]) < 0.05
    )


def _dead_fraction(reports: list[dict]) -> float:
    if not reports:
        return float("nan")
    n = len(reports)
    n_dead = sum(1 for r in reports if "dead" in r.get("flags", []))
    return n_dead / n if n > 0 else float("nan")


def _annotate_top_features(
    top_idx: np.ndarray, reports: list[dict]
) -> list[dict]:
    """Pair top feature indices with their feature_report.json metadata.

    For each ``feature_index`` in ``top_idx``, returns a dict with the
    interpretability-relevant fields from ``reports[feature_index]`` so a
    consumer can answer "what concept is feature 17 capturing?" without
    re-loading feature_report.json.

    Returns
    -------
    list[dict]
        One entry per top feature, in the order of ``top_idx``. Fields:
        ``feature_index``, ``flags``, ``mw_p_cognition``,
        ``ct_dominance`` (if present), ``dominant_cell_type`` (if present),
        ``fraction_active``. Missing keys yield ``None``.
    """
    out: list[dict] = []
    for idx in top_idx:
        i = int(idx)
        # Defensive: a malformed report.json could be shorter than fa_*.
        rep = reports[i] if i < len(reports) else {}
        out.append({
            "feature_index": i,
            "flags": list(rep.get("flags", [])),
            "mw_p_cognition": rep.get("mw_p_cognition"),
            "ct_dominance": rep.get("ct_dominance"),
            "dominant_cell_type": rep.get("dominant_cell_type"),
            "fraction_active": rep.get("fraction_active"),
        })
    return out


def _top10_decoder_cos_sim(
    W_dec_a: np.ndarray, fa_a: np.ndarray, reports_a: list[dict],
    W_dec_b: np.ndarray, fa_b: np.ndarray, reports_b: list[dict],
) -> dict:
    """Top-10 fraction-active features in each SAE; pairwise cos-sim matrix.

    Returns
    -------
    dict with
        ``cosine_matrix``                       — [K, K] cosine sim between
                                                  top-K columns (rows = trained,
                                                  cols = random).
        ``mean_abs_cosine_off_diag``            — mean of |cosine| off-diagonal
                                                  (coarse stat, kept for
                                                  reference).
        ``hungarian_mean_diagonal_cosine``      — canonical metric: average
                                                  diagonal cosine after
                                                  ``scipy.optimize.linear_sum_assignment(-cos)``
                                                  pairs each trained feature
                                                  with its best random match
                                                  (one-to-one). Per Orlov
                                                  §8.3 / Heap 2026.
        ``hungarian_assignment``                — list of (trained_rank,
                                                  random_rank) tuples produced
                                                  by linear_sum_assignment.
        ``top_features_trained_annotated``      — list of dicts joining each
                                                  ``top_features_trained`` index
                                                  to the corresponding row in
                                                  feature_report.json (flags +
                                                  mw_p_cognition + ct_dominance
                                                  + dominant_cell_type +
                                                  fraction_active).
        ``top_features_random_annotated``       — same, for the random run.
    """
    from scipy.optimize import linear_sum_assignment

    if W_dec_a.shape[0] != W_dec_b.shape[0]:
        raise ValueError(
            f"Decoder row dims differ: {W_dec_a.shape[0]} vs {W_dec_b.shape[0]}; "
            "the two SAEs must share input dim n."
        )
    K = min(10, fa_a.size, fa_b.size)
    # kind="stable" so ties break by original index — matches
    # captum_composite_attribution + gsea_from_captum convention.
    top_a = np.argsort(-fa_a, kind="stable")[:K]
    top_b = np.argsort(-fa_b, kind="stable")[:K]
    cols_a = W_dec_a[:, top_a]  # [n, K]
    cols_b = W_dec_b[:, top_b]  # [n, K]

    norm_a = np.linalg.norm(cols_a, axis=0, keepdims=True) + 1e-12
    norm_b = np.linalg.norm(cols_b, axis=0, keepdims=True) + 1e-12
    cos = (cols_a / norm_a).T @ (cols_b / norm_b)  # [K, K]

    # Off-diagonal mean absolute cosine — kept for reference.
    mask_off = ~np.eye(K, dtype=bool)
    mean_abs = float(np.abs(cos[mask_off]).mean()) if K > 1 else float("nan")

    # Hungarian-aligned mean diagonal cosine (canonical metric).
    # linear_sum_assignment minimises a cost matrix; we maximise cosine by
    # passing the negative. row_ind == arange(K) since cos is square.
    if K >= 1:
        row_ind, col_ind = linear_sum_assignment(-cos)
        hungarian_assignment = [
            [int(r), int(c)] for r, c in zip(row_ind, col_ind)
        ]
        hungarian_mean = float(cos[row_ind, col_ind].mean())
    else:
        hungarian_assignment = []
        hungarian_mean = float("nan")

    return {
        "cosine_matrix": cos.tolist(),
        "mean_abs_cosine_off_diag": mean_abs,
        "hungarian_mean_diagonal_cosine": hungarian_mean,
        "hungarian_assignment": hungarian_assignment,
        "K": int(K),
        "top_features_trained": [int(i) for i in top_a.tolist()],
        "top_features_random": [int(i) for i in top_b.tolist()],
        # Joined feature_report.json metadata so a consumer reading the
        # comparison.json alone can see *which concept* each top feature
        # captures (flags + dominant_cell_type + ct_dominance) instead of
        # having to re-load the per-run feature_report.json.
        "top_features_trained_annotated": _annotate_top_features(top_a, reports_a),
        "top_features_random_annotated": _annotate_top_features(top_b, reports_b),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "--trained-dir",
        required=True,
        help=(
            "Path to a trained-encoder SAE run directory containing "
            "reconstruction_metrics.json, feature_report.json, sae_model.npz."
        ),
    )
    p.add_argument(
        "--random-dir",
        required=True,
        help=(
            "Path to the matching random-encoder SAE run directory (same "
            "architecture / layer / expansion / k / seed)."
        ),
    )
    p.add_argument(
        "--out",
        default="outputs/canonical/sae/random_encoder_null/comparison.json",
        help="Output JSON path.",
    )
    p.add_argument(
        "--criterion-multiplier",
        type=float,
        default=1.5,
        help="Acceptance multiplier (default 1.5x per Orlov §8.3).",
    )
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    trained_dir = Path(args.trained_dir)
    if not trained_dir.is_absolute():
        trained_dir = _WORKTREE_ROOT / trained_dir
    random_dir = Path(args.random_dir)
    if not random_dir.is_absolute():
        random_dir = _WORKTREE_ROOT / random_dir
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = _WORKTREE_ROOT / out_path

    trained = _load_run(trained_dir)
    rand = _load_run(random_dir)

    # Sanity: same SAE config? Raise on mismatch — silently comparing
    # across different architectures / expansions / Ks invites apples-to-
    # oranges interpretation. The matched random null at
    # ``outputs/canonical/sae/random_encoder/batch_topk/fused/exp32_k64_seed0/``
    # was built specifically for this comparison.
    cfg_t = trained["metrics"]["config"]
    cfg_r = rand["metrics"]["config"]
    cfg_match_keys = ("architecture", "expansion", "k")
    cfg_mismatch = {
        k: (cfg_t.get(k), cfg_r.get(k))
        for k in cfg_match_keys
        if cfg_t.get(k) != cfg_r.get(k)
    }
    if cfg_mismatch:
        raise ValueError(
            f"Config mismatch between trained and random runs: {cfg_mismatch}. "
            "Run the matching random-encoder SAE at the trained best-config "
            "(architecture / expansion / k) before invoking this comparator. "
            "See `outputs/canonical/sae/random_encoder/batch_topk/fused/exp32_k64_seed0/` "
            "for the canonical fused null."
        )

    # Per-config metrics.
    interp_trained = _interpretable_fraction(trained["reports"])
    interp_random = _interpretable_fraction(rand["reports"])
    mw_count_trained = _mw_p_cog_lt_05_count(trained["reports"])
    mw_count_random = _mw_p_cog_lt_05_count(rand["reports"])
    dead_trained = _dead_fraction(trained["reports"])
    dead_random = _dead_fraction(rand["reports"])

    # Decoder-direction comparison.
    cos_pack = _top10_decoder_cos_sim(
        trained["W_dec"], trained["fraction_active"], trained["reports"],
        rand["W_dec"], rand["fraction_active"], rand["reports"],
    )

    # Acceptance criterion. Schema-clean output: always emit a numeric ratio
    # (NaN if undefined) and two booleans the consumer can dispatch on:
    #   random_produced_zero_interpretable      — distinguishes +inf branch
    #   trained_produced_zero_interpretable     — distinguishes "both dead"
    random_zero = interp_random == 0
    trained_zero = interp_trained == 0
    if not random_zero:
        ratio: float = interp_trained / interp_random
        passed = bool(ratio > args.criterion_multiplier)
    elif not trained_zero:
        # Random produced zero interpretable features → ratio undefined
        # (was +inf in old schema; prefer NaN + boolean flag for clarity).
        ratio = float("nan")
        passed = True
    else:
        ratio = float("nan")
        passed = False

    payload = {
        "trained_run": str(trained_dir),
        "random_run": str(random_dir),
        "trained_config": cfg_t,
        "random_config": cfg_r,
        "config_mismatch": cfg_mismatch,
        "metrics": {
            "trained": {
                "interpretable_candidate_fraction": interp_trained,
                "mw_p_cognition_lt_0.05_count": mw_count_trained,
                "dead_fraction": dead_trained,
                "n_features": len(trained["reports"]),
                "fve_full": trained["metrics"]["full"]["fve"],
                "l0_mean_full": trained["metrics"]["full"]["l0_mean"],
            },
            "random": {
                "interpretable_candidate_fraction": interp_random,
                "mw_p_cognition_lt_0.05_count": mw_count_random,
                "dead_fraction": dead_random,
                "n_features": len(rand["reports"]),
                "fve_full": rand["metrics"]["full"]["fve"],
                "l0_mean_full": rand["metrics"]["full"]["l0_mean"],
            },
        },
        "decoder_cos_sim_top10_pairs": cos_pack,
        "acceptance": {
            "criterion_multiplier": float(args.criterion_multiplier),
            # Always emit a numeric ratio (or NaN); consumers should
            # dispatch on the boolean flags below rather than the magnitude
            # when random_produced_zero_interpretable is True.
            "trained_over_random_interpretable_ratio": ratio,
            "pass_criterion": passed,
            "random_produced_zero_interpretable": bool(random_zero),
            "trained_produced_zero_interpretable": bool(trained_zero),
            "rule": (
                f"trained_interpretable_fraction > {args.criterion_multiplier}x "
                "random_interpretable_fraction"
            ),
        },
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, default=str))
    logger.info("Wrote %s", out_path)

    # Brief stdout summary.
    print("\n=== Trained vs Random Encoder SAE comparison ===")
    print(f"  Trained run : {trained_dir}")
    print(f"  Random run  : {random_dir}")
    print(f"  Config match: {cfg_match_keys}, mismatches: {cfg_mismatch or 'none'}")
    print()
    print("  Metric                                  Trained    Random")
    print(
        f"  interpretable_candidate_fraction        {interp_trained:>7.4f}    {interp_random:>7.4f}"
    )
    print(
        f"  mw_p_cognition<0.05 count               {mw_count_trained:>7d}    {mw_count_random:>7d}"
    )
    print(
        f"  dead_fraction                           {dead_trained:>7.4f}    {dead_random:>7.4f}"
    )
    print(
        f"  fve_full                                {trained['metrics']['full']['fve']:>7.4f}    {rand['metrics']['full']['fve']:>7.4f}"
    )
    print(
        f"  l0_mean_full                            {trained['metrics']['full']['l0_mean']:>7.2f}    {rand['metrics']['full']['l0_mean']:>7.2f}"
    )
    print()
    print(
        f"  decoder top-10 mean |cos| off-diag:        {cos_pack['mean_abs_cosine_off_diag']:.4f}"
    )
    print(
        f"  decoder top-10 Hungarian mean diag cos:    {cos_pack['hungarian_mean_diagonal_cosine']:.4f}"
    )
    print()
    if random_zero and not trained_zero:
        ratio_str = "n/a (random produced zero interpretable features)"
    elif np.isnan(ratio):
        ratio_str = "nan (both runs produced zero interpretable features)"
    else:
        ratio_str = f"{ratio:.3f}"
    print(
        f"  trained / random interpretable ratio:   {ratio_str}"
    )
    print(f"  >{args.criterion_multiplier}x criterion (Orlov §8.3): "
          f"{'PASS' if passed else 'FAIL'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
