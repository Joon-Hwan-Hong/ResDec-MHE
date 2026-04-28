"""Train a single SAE config on random-encoder activations (Heap-style null).

This is the SECOND half of the random-encoder null pipeline (design doc
§8.3, ``docs/plans/2026-04-28-sparse-autoencoder-design.md``). It does NOT
sweep — it matches the architecture/expansion/k of the trained-encoder
best-config (resolved by reading that config's ``reconstruction_metrics.json``)
and trains the same SAEConfig on the random-encoder activations produced
by ``extract_sae_activations_random.py``.

Outputs (mirroring the trained-encoder layout):

    <out-dir>/{architecture}/{layer}/exp{E}_k{K}_seed{S}/
        sae_model.npz
        reconstruction_metrics.json
        feature_report.json

Usage
-----
    PYTHONPATH=<worktree-root> \\
    CUDA_VISIBLE_DEVICES=1 \\
    uv run python scripts/resdec_mhe/interpretability/run_sae_random_null.py \\
        --activations outputs/redesign/sae/random_encoder/activations_attended_seed0.npz \\
        --config-match outputs/redesign/sae/topk/attended/exp8_k16_seed0/reconstruction_metrics.json \\
        --out-dir outputs/redesign/sae/random_encoder

Arguments
---------
    --activations <path>   Random-encoder activations npz from
                           extract_sae_activations_random.py.
    --config-match <path>  Path to a trained-encoder reconstruction_metrics.json
                           whose architecture / expansion / k / aux-K we will
                           match exactly.
    --out-dir <path>       Random-null SAE output root (created if missing).
    --metadata-path <path> ROSMAP metadata.csv (default: from canonical config).
    --seed <int>           SAE training seed (default: take from --config-match).
    --top-k-subjects <int> interpret_features top-K (default 20).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
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

from src.analysis.sae_io import (  # noqa: E402
    expand_fold_idx_to_rows,
    load_metadata_lookup,
    save_sae_model,
)
from src.analysis.sparse_autoencoder import (  # noqa: E402
    ActivationBundle,
    SAEConfig,
    evaluate_reconstruction,
    interpret_features,
    train_sae_batch_topk,
    train_sae_topk,
)
from src.utils.provenance import git_sha  # noqa: E402

logger = logging.getLogger(__name__)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "--activations",
        required=True,
        help="Random-encoder activations npz (from extract_sae_activations_random.py).",
    )
    p.add_argument(
        "--config-match",
        required=True,
        help=(
            "Path to a trained-encoder reconstruction_metrics.json. The "
            "SAEConfig (architecture, expansion, k, aux_lambda, aux_k, "
            "decoder_unit_norm, learning_rate, batch_size, n_steps, seed) "
            "is read from its 'config' block and used verbatim for the null. "
            "Legacy 'l1_lambda' keys (pre-2026-04-28 cleanup) are silently "
            "dropped."
        ),
    )
    p.add_argument(
        "--out-dir",
        default="outputs/redesign/sae/random_encoder",
        help="Random-null SAE output root.",
    )
    p.add_argument(
        "--metadata-path",
        default=None,
        help="ROSMAP metadata.csv. If unset, reads from canonical config.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optimizer seed override; default = take from --config-match.",
    )
    p.add_argument("--top-k-subjects", type=int, default=20)
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    # ─────────────────────────────────────────────────────────────────────
    # Resolve paths.
    # ─────────────────────────────────────────────────────────────────────
    activations_path = Path(args.activations)
    if not activations_path.is_absolute():
        activations_path = _WORKTREE_ROOT / activations_path
    config_match_path = Path(args.config_match)
    if not config_match_path.is_absolute():
        config_match_path = _WORKTREE_ROOT / config_match_path
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = _WORKTREE_ROOT / out_dir

    if not activations_path.exists():
        raise FileNotFoundError(f"Activations npz not found: {activations_path}")
    if not config_match_path.exists():
        raise FileNotFoundError(f"Config-match metrics file not found: {config_match_path}")

    # ─────────────────────────────────────────────────────────────────────
    # Load random-encoder activations.
    # ─────────────────────────────────────────────────────────────────────
    logger.info("Loading random-encoder activations from %s", activations_path)
    npz = np.load(activations_path, allow_pickle=True)
    activations = npz["activations"]
    subject_ids = npz["subject_ids"]
    fold_indices = npz["fold_indices"]
    is_val = npz["is_val"]
    cell_types = npz["cell_types"] if "cell_types" in npz.files else None
    layer = str(npz["layer"])

    bundle = ActivationBundle(
        activations=activations,
        subject_ids=subject_ids,
        fold_indices=fold_indices,
        is_val=is_val,
        cell_types=cell_types,
        layer=layer,
    )

    # Flatten per-(subject, CT) for "fused".
    if layer == "fused":
        N, C, n = activations.shape
        flat = activations.reshape(N * C, n)
    elif layer == "attended":
        flat = activations
        n = flat.shape[1]
    else:
        raise ValueError(f"Unknown layer in activations: {layer!r}")
    logger.info("Random activations layer=%s flat shape=%s (n=%d)", layer, flat.shape, n)

    # ─────────────────────────────────────────────────────────────────────
    # Load trained-encoder config block to match.
    # ─────────────────────────────────────────────────────────────────────
    matched = json.loads(config_match_path.read_text())
    cfg_block = matched["config"]
    matched_layer = matched.get("layer", layer)
    if matched_layer != layer:
        raise ValueError(
            f"Layer mismatch: --activations is {layer!r} but --config-match "
            f"was trained on {matched_layer!r}. Use a matching trained-encoder "
            "metrics file."
        )

    seed = int(args.seed) if args.seed is not None else int(cfg_block["seed"])
    # NOTE: ``l1_lambda`` was dropped from SAEConfig as part of the
    # 2026-04-28 architecture cleanup (the L1 architecture was never used);
    # we silently ignore it if present in legacy ``--config-match`` metrics
    # files so back-compatibility holds.
    config = SAEConfig(
        architecture=cfg_block["architecture"],
        expansion=int(cfg_block["expansion"]),
        k=int(cfg_block["k"]) if cfg_block.get("k") is not None else None,
        aux_lambda=float(cfg_block.get("aux_lambda", 1.0 / 32.0)),
        aux_k=int(cfg_block.get("aux_k", 256)),
        decoder_unit_norm=bool(cfg_block.get("decoder_unit_norm", True)),
        learning_rate=float(cfg_block.get("learning_rate", 1e-4)),
        batch_size=int(cfg_block.get("batch_size", 64)),
        n_steps=int(cfg_block.get("n_steps", 50_000)),
        seed=seed,
    )
    logger.info("Matched SAEConfig from %s: %s", config_match_path, asdict(config))

    # ─────────────────────────────────────────────────────────────────────
    # Train SAE on random-encoder activations.
    # ─────────────────────────────────────────────────────────────────────
    run_dir = (
        out_dir / config.architecture / layer
        / f"exp{config.expansion}_k{config.k}_seed{config.seed}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info("=== Training random-null SAE: %s ===", run_dir)

    t0 = time.time()
    if config.architecture == "topk":
        sae = train_sae_topk(flat, config)
    elif config.architecture == "batch_topk":
        sae = train_sae_batch_topk(flat, config)
    else:
        raise ValueError(f"Unsupported architecture: {config.architecture}")
    train_minutes = (time.time() - t0) / 60.0
    logger.info("Random-null SAE training done in %.2f min", train_minutes)

    # Persist SAE.
    save_sae_model(sae, run_dir / "sae_model.npz")
    logger.info("Wrote %s", run_dir / "sae_model.npz")

    # Reconstruction metrics: full + per-(notional) fold. The random-encoder
    # "fold" is constant (single forward pass), but we honour the schema so
    # downstream aggregators don't trip.
    full_metrics = evaluate_reconstruction(sae, flat)
    per_fold_metrics: dict[str, dict[str, float]] = {}
    fold_idx_flat = expand_fold_idx_to_rows(
        fold_indices, layer,
        n_celltypes=int(activations.shape[1]) if layer == "fused" else None,
    )
    for f in sorted(set(fold_idx_flat.tolist())):
        mask = (fold_idx_flat == f)
        per_fold_metrics[f"fold{f}"] = evaluate_reconstruction(sae, flat[mask])

    metrics_payload = {
        "config": asdict(config),
        "layer": layer,
        "n_train_rows": int(flat.shape[0]),
        "full": full_metrics,
        "per_fold": per_fold_metrics,
        "train_minutes": train_minutes,
        "git_commit": git_sha(_WORKTREE_ROOT),
        "encoder": "random_init",
        "config_match_source": str(config_match_path),
    }
    metrics_path = run_dir / "reconstruction_metrics.json"
    metrics_path.write_text(json.dumps(metrics_payload, indent=2, default=str))
    logger.info("Wrote %s (full FVE=%.4f)", metrics_path, full_metrics["fve"])

    # Per-feature interpretability report.
    metadata_path = args.metadata_path
    if metadata_path is None:
        from omegaconf import OmegaConf

        cfg = OmegaConf.merge(
            OmegaConf.load(_WORKTREE_ROOT / "configs" / "default.yaml"),
            OmegaConf.load(_WORKTREE_ROOT / "configs" / "resdec_mhe" / "canonical.yaml"),
        )
        metadata_path = Path(cfg.data.metadata_path) / "metadata.csv"
        if not metadata_path.is_absolute():
            metadata_path = _WORKTREE_ROOT / metadata_path
    metadata_path = Path(metadata_path)
    metadata = load_metadata_lookup(metadata_path, subject_ids)

    reports = interpret_features(
        sae=sae, bundle=bundle, metadata=metadata,
        top_k_subjects=int(args.top_k_subjects),
    )
    serializable_reports = []
    for r in reports:
        rr = dict(r)
        rr["flags"] = sorted(list(rr.get("flags", set())))
        serializable_reports.append(rr)
    report_path = run_dir / "feature_report.json"
    report_path.write_text(json.dumps(serializable_reports, indent=2, default=str))
    logger.info(
        "Wrote %s (%d features, %d interpretable_candidates)",
        report_path,
        len(serializable_reports),
        sum(1 for r in serializable_reports if "interpretable_candidate" in r["flags"]),
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
