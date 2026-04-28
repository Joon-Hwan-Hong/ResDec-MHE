"""Train one SAE config on previously-extracted ResDec-MHE encoder activations.

Loads ``activations_{layer}_all_folds.npz`` (written by
``extract_sae_activations.py``), fits a TopK or Batch-TopK SAE per Orlov 2026 /
Bussmann 2024 / Gao 2024, computes reconstruction metrics (FVE, L0, dead
fraction) on (a) the full union and (b) each fold's held-out activations as a
cross-fold-stability sanity check, then runs ``interpret_features`` to produce
a per-feature interpretability report.

Outputs a single run directory ``<out-root>/{architecture}/{layer}/exp{exp}_k{k}_seed{seed}/``
with:
  - ``sae_model.npz``               — fitted SAE weights + activation_stats
  - ``reconstruction_metrics.json`` — full + per-fold FVE / L0 / dead_fraction
  - ``feature_report.json``         — per-feature top subjects, top CTs, MW p-values, flags

Usage
-----
    PYTHONPATH=<worktree-root> \\
    CUDA_VISIBLE_DEVICES=0 \\
    uv run python scripts/resdec_mhe/interpretability/run_sae_train.py \\
        --activations-dir outputs/redesign/sae \\
        --layer attended \\
        --architecture batch_topk \\
        --expansion 16 --k 8 --seed 0 \\
        --n-steps 50000 --batch-size 64 --learning-rate 1e-4 \\
        --out-root outputs/redesign/sae

Arguments
---------
    --activations-dir <path>     Directory holding ``activations_{layer}_all_folds.npz``.
    --metadata-path <path>       ROSMAP metadata.csv (default: from canonical config).
    --layer <{attended, fused}>  Which extracted layer to train on.
    --architecture <{topk, batch_topk}>
    --expansion <{8, 16, 32}>    Dictionary expansion (m / n).
    --k <int>                    Active features per sample (TopK / batch K-th).
    --aux-lambda <float>         Auxiliary-K loss weight (default 1/32 per Gao 2024).
    --aux-k <int>                Number of dead features in aux-K loss (default 256).
    --seed <int>                 Optimizer seed (default 0).
    --n-steps <int>              Optimizer steps (default 50000).
    --batch-size <int>           Optimizer batch size (default 64).
    --learning-rate <float>      Adam lr (default 1e-4).
    --out-root <path>            Per-run output dir root.
    --top-k-subjects <int>       interpret_features top-K (default 20).
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
        "--activations-dir",
        default="outputs/redesign/sae",
        help="Directory holding activations_{layer}_all_folds.npz.",
    )
    p.add_argument(
        "--metadata-path",
        default=None,
        help=(
            "Path to ROSMAP metadata.csv. If unset, reads from the canonical "
            "config's data.metadata_path."
        ),
    )
    p.add_argument("--layer", choices=["attended", "fused"], required=True)
    p.add_argument("--architecture", choices=["topk", "batch_topk"], required=True)
    p.add_argument("--expansion", type=int, choices=[8, 16, 32], required=True)
    p.add_argument("--k", type=int, required=True)
    p.add_argument("--aux-lambda", type=float, default=1.0 / 32.0)
    p.add_argument("--aux-k", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-steps", type=int, default=50_000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument(
        "--out-root",
        default="outputs/redesign/sae",
        help="Output root for SAE runs.",
    )
    p.add_argument("--top-k-subjects", type=int, default=20)
    p.add_argument(
        "--decoder-unit-norm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Apply unit-norm constraint to decoder columns "
            "(Bussmann 2024 standard); pass --no-decoder-unit-norm to disable."
        ),
    )
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    activations_dir = (
        Path(args.activations_dir) if Path(args.activations_dir).is_absolute()
        else _WORKTREE_ROOT / args.activations_dir
    )
    out_root = (
        Path(args.out_root) if Path(args.out_root).is_absolute()
        else _WORKTREE_ROOT / args.out_root
    )

    bundle_npz = activations_dir / f"activations_{args.layer}_all_folds.npz"
    if not bundle_npz.exists():
        raise FileNotFoundError(
            f"Expected union activations file at {bundle_npz}. "
            "Run extract_sae_activations.py first."
        )
    logger.info("Loading activations from %s", bundle_npz)
    npz = np.load(bundle_npz, allow_pickle=True)
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
        layer=args.layer,
    )

    # Flatten per-(subject, CT) for "fused".
    if layer == "fused":
        N, C, n = activations.shape
        flat = activations.reshape(N * C, n)
    else:
        flat = activations
        n = flat.shape[1]
    logger.info(
        "layer=%s flat shape=%s (n=%d)", layer, flat.shape, n,
    )

    config = SAEConfig(
        architecture=args.architecture,
        expansion=int(args.expansion),
        k=int(args.k),
        aux_lambda=float(args.aux_lambda),
        aux_k=int(args.aux_k),
        decoder_unit_norm=bool(args.decoder_unit_norm),
        learning_rate=float(args.learning_rate),
        batch_size=int(args.batch_size),
        n_steps=int(args.n_steps),
        seed=int(args.seed),
    )

    run_dir = (
        out_root / args.architecture / args.layer
        / f"exp{args.expansion}_k{args.k}_seed{args.seed}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info("=== Training SAE: %s ===", run_dir)

    t0 = time.time()
    if config.architecture == "topk":
        sae = train_sae_topk(flat, config)
    elif config.architecture == "batch_topk":
        sae = train_sae_batch_topk(flat, config)
    else:
        raise ValueError(f"Unsupported architecture: {config.architecture}")
    train_minutes = (time.time() - t0) / 60.0
    logger.info("Training done in %.2f min", train_minutes)

    # Persist SAE.
    save_sae_model(sae, run_dir / "sae_model.npz")
    logger.info("Wrote %s", run_dir / "sae_model.npz")

    # Reconstruction metrics: full + per-fold (cross-fold-stability sanity).
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

    metadata = load_metadata_lookup(Path(metadata_path), subject_ids)

    reports = interpret_features(
        sae=sae, bundle=bundle, metadata=metadata,
        top_k_subjects=int(args.top_k_subjects),
    )
    # Convert sets to lists for JSON serialisation.
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
