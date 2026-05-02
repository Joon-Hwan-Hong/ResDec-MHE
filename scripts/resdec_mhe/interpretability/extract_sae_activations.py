"""Extract per-(subject, layer) hidden-state activations for SAE training.

For each of the 5 canonical ResDec-MHE folds, loads the max-R² checkpoint,
runs forward over both train + val splits with ``return_embeddings=True``,
and persists the requested layer activations (``attended`` ``[B, 64]`` and/or
``fused`` ``[B, 31, 64]``) per fold + a combined union .npz.

Outputs (default ``outputs/canonical/sae/``):
  - activations_attended_fold{f}.npz         per-fold ``attended``
  - activations_attended_all_folds.npz       union (~2580 rows for ``attended``)
  - activations_fused_fold{f}.npz            per-fold ``fused``
  - activations_fused_all_folds.npz          union (~2580 rows × 31 CTs)

Usage
-----
    PYTHONPATH=<worktree-root> \\
    CUDA_VISIBLE_DEVICES=0 \\
    uv run python scripts/resdec_mhe/interpretability/extract_sae_activations.py \\
        --pred-root outputs/canonical/p5_canonical_seed42 \\
        --out-dir outputs/canonical/sae \\
        --layers attended fused

Arguments
---------
    --config <path>      Phase YAML merged on top of configs/default.yaml
                         (default: ``configs/resdec_mhe/canonical.yaml``).
    --pred-root <path>   Per-fold output dir with ``fold{0..4}/checkpoints/best-*.ckpt``.
    --splits-path <path> Splits JSON (default: ``outputs/splits.json``).
    --out-dir <path>     Output directory (created if missing).
    --layers <list>      Which layers to extract; choices ``attended``, ``fused``.
    --device             Torch device (default: ``cuda``).
    --batch-size         Forward-pass batch size (default: 32).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

_WORKTREE_ROOT = Path(__file__).resolve().parents[3]
if not (_WORKTREE_ROOT / "src").is_dir():
    raise RuntimeError(
        f"sys.path bootstrap failed: {_WORKTREE_ROOT}/src not found; "
        "set PYTHONPATH=<worktree-root>."
    )
sys.path.insert(0, str(_WORKTREE_ROOT))

from src.analysis.sparse_autoencoder import extract_activations  # noqa: E402
from src.utils.provenance import pick_max_r2_ckpt  # noqa: E402

logger = logging.getLogger(__name__)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "--config",
        default="configs/resdec_mhe/canonical.yaml",
        help="Phase YAML merged on top of configs/default.yaml.",
    )
    p.add_argument(
        "--pred-root",
        default="outputs/canonical/p5_canonical_seed42",
        help="Per-fold output dir with fold{0..4}/checkpoints/best-*.ckpt.",
    )
    p.add_argument(
        "--splits-path",
        default="outputs/splits.json",
        help="Splits JSON path.",
    )
    p.add_argument(
        "--out-dir",
        default="outputs/canonical/sae",
        help="Destination directory for activation .npz files.",
    )
    p.add_argument(
        "--layers",
        nargs="+",
        choices=["attended", "fused"],
        default=["attended", "fused"],
        help="Layers to extract.",
    )
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--n-folds", type=int, default=5)
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    pred_root = (
        Path(args.pred_root) if Path(args.pred_root).is_absolute()
        else _WORKTREE_ROOT / args.pred_root
    )
    out_dir = (
        Path(args.out_dir) if Path(args.out_dir).is_absolute()
        else _WORKTREE_ROOT / args.out_dir
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve per-fold max-R² checkpoint paths up front.
    checkpoint_paths: list[Path] = []
    for fold in range(args.n_folds):
        ckpt_dir = pred_root / f"fold{fold}" / "checkpoints"
        ckpt = pick_max_r2_ckpt(ckpt_dir)
        logger.info("fold %d: %s", fold, ckpt)
        checkpoint_paths.append(ckpt)

    for layer in args.layers:
        logger.info("=== Extracting layer=%s to %s ===", layer, out_dir)
        bundle = extract_activations(
            checkpoint_paths=checkpoint_paths,
            layer=layer,
            output_dir=out_dir,
            device=args.device,
            batch_size=int(args.batch_size),
        )
        n_subj = int(bundle.activations.shape[0])
        n_folds = len({int(f) for f in bundle.fold_indices.tolist()})
        n_val = int(bundle.is_val.sum())
        logger.info(
            "layer=%s: extracted shape=%s, %d folds, %d val rows",
            layer, bundle.activations.shape, n_folds, n_val,
        )
        # Persist a small summary alongside the npz so a user can verify
        # n_subj / n_val / n_folds / layer post-hoc without re-running.
        summary_path = out_dir / f"extract_summary_{layer}.json"
        summary_path.write_text(
            json.dumps(
                {
                    "layer": layer,
                    "n_subjects": n_subj,
                    "n_val_rows": n_val,
                    "n_folds_seen": n_folds,
                    "activation_shape": list(bundle.activations.shape),
                    "checkpoint_paths": [str(p) for p in checkpoint_paths],
                    "out_dir": str(out_dir),
                },
                indent=2,
            )
        )
        logger.info("Wrote %s", summary_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
