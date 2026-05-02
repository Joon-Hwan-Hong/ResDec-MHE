"""Single-process SAE sweep driver (F12 alternative to ``run_sae_sweep.sh``).

The shell driver in this directory spawns ``uv run python`` once per
hyperparameter combination — 60 times for the canonical 2 × 2 × 3 × 5 grid.
Each spawn re-pays Python interpreter startup, ``import torch`` (~few seconds),
``OmegaConf.load`` of default + canonical yaml, and the activation-bundle
``np.load``. Aggregated this is ~5-10 minutes of wasted wall time per shard.

This module provides an in-process equivalent: imports the SAE training
helpers once, loads the activation bundle once, then iterates the same grid
calling :func:`_train_one_config` per cell. Identical numerics to running
``run_sae_train.py`` per config — the underlying ``train_sae_topk`` /
``train_sae_batch_topk`` calls are unchanged.

Per F12 spec: both ``run_sae_sweep.sh`` and this Python driver are kept for
idempotency. The .sh remains the canonical CI-friendly entry point; this
file is a fast-path replacement when the user wants minimum wall time.

Usage
-----
    PYTHONPATH=<worktree-root> \\
    CUDA_VISIBLE_DEVICES=0 \\
    uv run python scripts/resdec_mhe/run_sae_sweep_inline.py \\
        --activations-dir outputs/canonical/sae \\
        --out-root outputs/canonical/sae \\
        --gpu-index 0 --num-gpus 2

Env vars (optional, mirroring the .sh defaults):
    METADATA_PATH, N_STEPS, BATCH_SIZE, LEARNING_RATE, AUX_LAMBDA, AUX_K, SEED
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

_WORKTREE_ROOT = Path(__file__).resolve().parents[2]
if str(_WORKTREE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKTREE_ROOT))

from src.analysis.sae_io import (  # noqa: E402
    expand_fold_idx_to_rows,
    load_metadata_lookup,
    save_sae_model,
)
from src.analysis.sparse_autoencoder import (  # noqa: E402
    ActivationBundle,
    SAEConfig,
    evaluate_reconstruction_with_cached_codes,
    interpret_features,
    reconstruction_metrics_from_slice,
    train_sae_batch_topk,
    train_sae_topk,
)
from src.utils.provenance import git_sha  # noqa: E402

logger = logging.getLogger(__name__)

# Sweep grid axes. KEEP IN LOCKSTEP with the bash arrays in
# ``run_sae_sweep.sh:87-90`` (canonical 60-config sweep). The smaller-M
# variant (``run_sae_sweep_smaller_m.sh:91-95``) drives its own axes via
# env-var read into bash arrays. Any change to ARCHITECTURES / LAYERS /
# EXPANSIONS / K_VALUES MUST also be made in run_sae_sweep.sh (B-SI1/B-SS1).
ARCHITECTURES = ("topk", "batch_topk")
LAYERS = ("attended", "fused")
EXPANSIONS = (8, 16, 32)
K_VALUES = (4, 8, 16, 32, 64)


def _load_bundle(activations_dir: Path, layer: str) -> tuple[
    ActivationBundle, np.ndarray, np.ndarray, np.ndarray, str,
]:
    bundle_npz = activations_dir / f"activations_{layer}_all_folds.npz"
    if not bundle_npz.exists():
        raise FileNotFoundError(
            f"Expected union activations file at {bundle_npz}. "
            "Run extract_sae_activations.py first."
        )
    npz = np.load(bundle_npz, allow_pickle=True)
    activations = npz["activations"]
    subject_ids = npz["subject_ids"]
    fold_indices = npz["fold_indices"]
    is_val = npz["is_val"]
    cell_types = npz["cell_types"] if "cell_types" in npz.files else None
    layer_str = str(npz["layer"])
    bundle = ActivationBundle(
        activations=activations,
        subject_ids=subject_ids,
        fold_indices=fold_indices,
        is_val=is_val,
        cell_types=cell_types,
        layer=layer_str,
    )
    if layer_str == "fused":
        N, C, n = activations.shape
        flat = activations.reshape(N * C, n)
    else:
        flat = activations
    return bundle, flat, subject_ids, fold_indices, layer_str


def _train_one_config(
    bundle: ActivationBundle,
    flat: np.ndarray,
    subject_ids: np.ndarray,
    fold_indices: np.ndarray,
    layer: str,
    architecture: str,
    expansion: int,
    k: int,
    seed: int,
    n_steps: int,
    batch_size: int,
    learning_rate: float,
    aux_lambda: float,
    aux_k: int,
    metadata: dict[str, np.ndarray],
    top_k_subjects: int,
    out_root: Path,
) -> bool:
    config = SAEConfig(
        architecture=architecture,
        expansion=int(expansion),
        k=int(k),
        aux_lambda=float(aux_lambda),
        aux_k=int(aux_k),
        decoder_unit_norm=True,
        learning_rate=float(learning_rate),
        batch_size=int(batch_size),
        n_steps=int(n_steps),
        seed=int(seed),
    )
    run_dir = (
        out_root / architecture / layer
        / f"exp{expansion}_k{k}_seed{seed}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_file = run_dir / "reconstruction_metrics.json"
    if metrics_file.exists():
        logger.info("SKIP existing %s", run_dir)
        return False

    logger.info("=== %s ===", run_dir)
    t0 = time.time()
    if architecture == "topk":
        sae = train_sae_topk(flat, config)
    else:
        sae = train_sae_batch_topk(flat, config)
    train_minutes = (time.time() - t0) / 60.0
    save_sae_model(sae, run_dir / "sae_model.npz")

    full_metrics, h_full, x_hat_full = evaluate_reconstruction_with_cached_codes(
        sae, flat,
    )
    per_fold_metrics: dict[str, dict[str, float]] = {}
    fold_idx_flat = expand_fold_idx_to_rows(
        fold_indices, layer,
        n_celltypes=int(bundle.activations.shape[1]) if layer == "fused" else None,
    )
    for f in sorted(set(fold_idx_flat.tolist())):
        mask = (fold_idx_flat == f)
        per_fold_metrics[f"fold{f}"] = reconstruction_metrics_from_slice(
            flat, h_full, x_hat_full, mask,
        )
    metrics_payload = {
        "config": asdict(config),
        "layer": layer,
        "n_train_rows": int(flat.shape[0]),
        "full": full_metrics,
        "per_fold": per_fold_metrics,
        "train_minutes": train_minutes,
        "git_commit": git_sha(_WORKTREE_ROOT),
    }
    metrics_file.write_text(json.dumps(metrics_payload, indent=2, default=str))

    reports = interpret_features(
        sae=sae, bundle=bundle, metadata=metadata,
        top_k_subjects=int(top_k_subjects),
    )
    serializable = []
    for r in reports:
        rr = dict(r)
        rr["flags"] = sorted(list(rr.get("flags", set())))
        serializable.append(rr)
    (run_dir / "feature_report.json").write_text(
        json.dumps(serializable, indent=2, default=str),
    )
    logger.info("done %s (%.2f min)", run_dir, train_minutes)
    return True


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "--activations-dir",
        default=str(_WORKTREE_ROOT / "outputs" / "canonical" / "sae"),
    )
    p.add_argument(
        "--out-root",
        default=str(_WORKTREE_ROOT / "outputs" / "canonical" / "sae"),
    )
    p.add_argument(
        "--metadata-path",
        default=os.environ.get(
            "METADATA_PATH",
            str(_WORKTREE_ROOT / "data" / "metadata_ROSMAP" / "metadata.csv"),
        ),
    )
    p.add_argument("--gpu-index", type=int,
                   default=int(os.environ.get("GPU_INDEX", "0")))
    p.add_argument("--num-gpus", type=int,
                   default=int(os.environ.get("NUM_GPUS", "1")))
    p.add_argument("--seed", type=int,
                   default=int(os.environ.get("SEED", "0")))
    p.add_argument("--n-steps", type=int,
                   default=int(os.environ.get("N_STEPS", "50000")))
    p.add_argument("--batch-size", type=int,
                   default=int(os.environ.get("BATCH_SIZE", "64")))
    p.add_argument("--learning-rate", type=float,
                   default=float(os.environ.get("LEARNING_RATE", "1e-4")))
    p.add_argument("--aux-lambda", type=float,
                   default=float(os.environ.get("AUX_LAMBDA", str(1.0 / 32.0))))
    p.add_argument("--aux-k", type=int,
                   default=int(os.environ.get("AUX_K", "256")))
    p.add_argument("--top-k-subjects", type=int, default=20)
    args = p.parse_args()

    if args.gpu_index >= args.num_gpus:
        raise SystemExit(
            f"--gpu-index ({args.gpu_index}) must be < --num-gpus ({args.num_gpus})"
        )

    # GPU-assignment invariant: --gpu-index / --num-gpus only control which
    # CONFIGS this shard runs (`idx % num_gpus == gpu_index`). The actual
    # physical GPU is determined by the parent process's CUDA_VISIBLE_DEVICES
    # mask. Recommended pattern (from run_sae_sweep.sh launch comments):
    #   tmux new -s sae_g0
    #   CUDA_VISIBLE_DEVICES=0 GPU_INDEX=0 NUM_GPUS=2 \
    #     uv run python scripts/resdec_mhe/run_sae_sweep_inline.py
    #   tmux new -s sae_g1
    #   CUDA_VISIBLE_DEVICES=1 GPU_INDEX=1 NUM_GPUS=2 \
    #     uv run python scripts/resdec_mhe/run_sae_sweep_inline.py
    # Each shell sees its physical GPU as logical 0; the in-process torch
    # ops below default to that single visible device.

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    activations_dir = Path(args.activations_dir).resolve()
    out_root = Path(args.out_root).resolve()
    metadata_path = Path(args.metadata_path).resolve()

    # Load both layers' bundles up front.
    bundles_by_layer: dict[str, tuple] = {}
    for layer in LAYERS:
        bundles_by_layer[layer] = _load_bundle(activations_dir, layer)

    # Load metadata once per (sids, csv) pair.
    metadata_by_layer: dict[str, dict[str, np.ndarray]] = {}
    for layer, (_bundle, _flat, sids, _fold, _layer_str) in bundles_by_layer.items():
        metadata_by_layer[layer] = load_metadata_lookup(metadata_path, sids)

    n_total = (
        len(ARCHITECTURES) * len(LAYERS) * len(EXPANSIONS) * len(K_VALUES)
    )
    n_in_shard = 0
    n_run = 0
    n_skip = 0
    n_fail = 0
    idx = 0
    for arch in ARCHITECTURES:
        for layer in LAYERS:
            bundle, flat, sids, fold_indices, layer_str = bundles_by_layer[layer]
            metadata = metadata_by_layer[layer]
            for exp in EXPANSIONS:
                for k in K_VALUES:
                    if idx % args.num_gpus != args.gpu_index:
                        idx += 1
                        continue
                    idx += 1
                    n_in_shard += 1
                    try:
                        ran = _train_one_config(
                            bundle=bundle,
                            flat=flat,
                            subject_ids=sids,
                            fold_indices=fold_indices,
                            layer=layer_str,
                            architecture=arch,
                            expansion=exp,
                            k=k,
                            seed=args.seed,
                            n_steps=args.n_steps,
                            batch_size=args.batch_size,
                            learning_rate=args.learning_rate,
                            aux_lambda=args.aux_lambda,
                            aux_k=args.aux_k,
                            metadata=metadata,
                            top_k_subjects=args.top_k_subjects,
                            out_root=out_root,
                        )
                        if ran:
                            n_run += 1
                        else:
                            n_skip += 1
                    except Exception as exc:  # noqa: BLE001
                        # Deliberately broad: a single config failure must not
                        # abort the rest of the sweep — per-config exceptions
                        # are logged with full traceback (logger.exception)
                        # and the sweep continues. Mirrors the set +e/-e
                        # toggle in run_sae_sweep.sh:135-154.
                        n_fail += 1
                        logger.exception(
                            "config %s/%s/exp%d_k%d failed: %s",
                            arch, layer, exp, k, exc,
                        )
    logger.info(
        "DONE: total=%d in_shard=%d ran=%d skipped=%d failed=%d",
        n_total, n_in_shard, n_run, n_skip, n_fail,
    )
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
