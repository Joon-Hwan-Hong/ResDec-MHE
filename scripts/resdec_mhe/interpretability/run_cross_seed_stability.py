"""Compute SAE cross-seed decoder stability per Paulo & Belrose 2025 (Orlov §8.4).

For S SAE models trained on the same activations with different random seeds,
compute decoder column-cosine similarity matrices for every (s, s') pair, then
report the per-feature stability mask and ``stable_fraction`` scalar at a
0.7 cosine threshold (Paulo & Belrose 2025).

Outputs
-------
``<out-dir>/cross_seed_summary.json``
    Includes ``stable_fraction``, ``S``, ``m`` (dictionary size),
    ``cosine_threshold``, per-pair off-diagonal cosine summary statistics
    (mean / median / min), per-feature stability counts, and per-seed
    metadata (seed value, source dir).
``<out-dir>/decoder_cosine_matrices.npz``
    Saves the [S, S, m, m] stack of decoder cosine similarity matrices and
    the [m] boolean per-feature stability mask.

Usage
-----
    PYTHONPATH=<worktree-root> uv run python \\
        scripts/resdec_mhe/interpretability/run_cross_seed_stability.py \\
        --run-dirs outputs/redesign/sae/batch_topk/fused/exp32_k64_seed0 \\
                   outputs/redesign/sae/batch_topk/fused/exp32_k64_seed1 \\
                   outputs/redesign/sae/batch_topk/fused/exp32_k64_seed2 \\
        --out-dir outputs/redesign/sae/cross_seed_stability \\
        --cosine-threshold 0.7
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

_WORKTREE_ROOT = Path(__file__).resolve().parents[3]
if not (_WORKTREE_ROOT / "src").is_dir():
    raise RuntimeError(
        f"sys.path bootstrap failed: {_WORKTREE_ROOT}/src not found; "
        "set PYTHONPATH=<worktree-root>."
    )
sys.path.insert(0, str(_WORKTREE_ROOT))

from src.analysis.sparse_autoencoder import (  # noqa: E402
    SAEConfig,
    SAEModel,
    cross_seed_stability,
)
from src.utils.provenance import git_sha  # noqa: E402

logger = logging.getLogger(__name__)


def _load_sae_model(run_dir: Path) -> SAEModel:
    """Load a trained SAEModel from ``<run_dir>/sae_model.npz``.

    Mirrors the persistence format used by ``run_sae_train.py``: ``W_enc``,
    ``W_dec``, ``b_enc``, ``b_dec``, plus per-feature ``stat_*`` arrays and
    a JSON-serialised ``config_json``.

    Drops legacy keys (e.g., ``l1_lambda`` from the pre-2026-04-28 architecture
    cleanup) so older runs continue to load after the dataclass surface was
    trimmed.
    """
    npz_path = run_dir / "sae_model.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"SAE model file missing: {npz_path}")
    npz = np.load(npz_path, allow_pickle=True)
    cfg_dict = json.loads(str(npz["config_json"]))
    # Strip dropped fields to maintain back-compatibility with pre-cleanup
    # serialised SAEs.
    cfg_dict.pop("l1_lambda", None)
    config = SAEConfig(**cfg_dict)
    activation_stats: dict[str, np.ndarray] = {}
    for key in ("mean", "std", "fraction_active", "is_dead", "threshold"):
        np_key = f"stat_{key}"
        if np_key in npz.files and np.asarray(npz[np_key]).size > 0:
            activation_stats[key] = np.asarray(npz[np_key])
    return SAEModel(
        W_enc=np.asarray(npz["W_enc"]),
        b_enc=np.asarray(npz["b_enc"]),
        W_dec=np.asarray(npz["W_dec"]),
        b_dec=np.asarray(npz["b_dec"]),
        config=config,
        activation_stats=activation_stats,
    )


def _off_diagonal_summary(C: np.ndarray) -> dict[str, float]:
    """Summary statistics over off-diagonal entries of an m×m cosine matrix."""
    m = C.shape[0]
    if C.shape[1] != m:
        raise ValueError(f"Expected square matrix; got shape {C.shape}")
    mask = ~np.eye(m, dtype=bool)
    off = C[mask]
    return {
        "mean": float(np.mean(off)),
        "median": float(np.median(off)),
        "min": float(np.min(off)),
        "max": float(np.max(off)),
        "std": float(np.std(off)),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "--run-dirs",
        nargs="+",
        required=True,
        help=(
            "List of SAE run directories (each containing sae_model.npz). "
            "Order is preserved for matrix indexing — pass seed-0 first."
        ),
    )
    p.add_argument(
        "--out-dir",
        required=True,
        help="Directory to write cross_seed_summary.json + decoder_cosine_matrices.npz.",
    )
    p.add_argument(
        "--cosine-threshold",
        type=float,
        default=0.7,
        help="Best-match cosine threshold for stable-feature counting (Paulo & Belrose 2025).",
    )
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
    )

    run_dirs = []
    for d in args.run_dirs:
        path = Path(d)
        if not path.is_absolute():
            path = _WORKTREE_ROOT / path
        run_dirs.append(path)

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = _WORKTREE_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading %d SAE models", len(run_dirs))
    sae_models: list[SAEModel] = []
    seed_metadata: list[dict] = []
    for rd in run_dirs:
        sae = _load_sae_model(rd)
        sae_models.append(sae)
        logger.info(
            "  %s: seed=%d arch=%s exp=%d k=%s W_dec.shape=%s",
            rd.name, sae.config.seed, sae.config.architecture,
            sae.config.expansion, sae.config.k, sae.W_dec.shape,
        )
        seed_metadata.append({
            "run_dir": str(rd.relative_to(_WORKTREE_ROOT)),
            "seed": int(sae.config.seed),
            "config": asdict(sae.config),
        })

    logger.info(
        "Computing cross-seed stability (threshold=%.2f)", args.cosine_threshold,
    )
    result = cross_seed_stability(
        sae_models, cosine_threshold=float(args.cosine_threshold),
    )
    cosine_matrices = result["cosine_matrices"]   # [S, S, m, m]
    per_feat = result["per_feature_stability"]    # [m] bool — Hungarian
    per_feat_argmax = result["per_feature_stability_argmax"]  # [m] bool — legacy
    stable_fraction = float(result["stable_fraction"])
    stable_fraction_argmax = float(result["stable_fraction_argmax"])

    S = cosine_matrices.shape[0]
    m = cosine_matrices.shape[2]

    # Per-pair off-diagonal stats (excluding self-pair s == s' which is identity)
    pair_stats: dict[str, dict[str, float]] = {}
    for s in range(S):
        for sp in range(S):
            if s == sp:
                continue
            key = f"seed{seed_metadata[s]['seed']}_vs_seed{seed_metadata[sp]['seed']}"
            pair_stats[key] = _off_diagonal_summary(cosine_matrices[s, sp])

    # Aggregate cosine stats across all unordered pairs (s != s').
    all_pair_offdiag_means = [v["mean"] for v in pair_stats.values()]
    all_pair_offdiag_medians = [v["median"] for v in pair_stats.values()]
    all_pair_offdiag_mins = [v["min"] for v in pair_stats.values()]

    summary_payload = {
        "git_commit": git_sha(_WORKTREE_ROOT),
        "S": int(S),
        "m": int(m),
        "cosine_threshold": float(args.cosine_threshold),
        "stable_fraction": stable_fraction,
        "stable_fraction_argmax": stable_fraction_argmax,
        "n_stable_features": int(per_feat.sum()),
        "n_stable_features_argmax": int(per_feat_argmax.sum()),
        "metric_definitions": {
            "stable_fraction": (
                "Canonical Paulo & Belrose 2025 / Orlov §8.4: bipartite "
                "(Hungarian) matching of seed-0 features to each other seed's "
                "features, then fraction whose assigned-pair cosine "
                f">= {float(args.cosine_threshold)} across every other seed."
            ),
            "stable_fraction_argmax": (
                "Legacy argmax-per-feature definition (informational only). "
                "Allows multiple seed-0 features to collapse onto the same "
                "seed-sp target, which can inflate the reported fraction."
            ),
        },
        "seeds": seed_metadata,
        "pair_offdiag_stats": pair_stats,
        "aggregate_offdiag_stats": {
            "mean_of_pair_means": float(np.mean(all_pair_offdiag_means)),
            "median_of_pair_medians": float(np.median(all_pair_offdiag_medians)),
            "min_of_pair_mins": float(np.min(all_pair_offdiag_mins)),
        },
    }
    summary_path = out_dir / "cross_seed_summary.json"
    summary_path.write_text(json.dumps(summary_payload, indent=2))
    logger.info(
        "Wrote %s (Hungarian stable_fraction=%.4f, argmax stable_fraction=%.4f, m=%d, S=%d)",
        summary_path, stable_fraction, stable_fraction_argmax, m, S,
    )

    matrices_path = out_dir / "decoder_cosine_matrices.npz"
    np.savez(
        matrices_path,
        cosine_matrices=cosine_matrices,
        per_feature_stability=per_feat,
        per_feature_stability_argmax=per_feat_argmax,
        seed_order=np.array([sm["seed"] for sm in seed_metadata], dtype=np.int64),
    )
    logger.info(
        "Wrote %s (cosine_matrices.shape=%s)",
        matrices_path, cosine_matrices.shape,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
