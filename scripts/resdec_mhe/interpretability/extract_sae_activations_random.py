"""Extract activations from a randomly-initialized ResDec-MHE encoder (Heap-style null).

Per the SAE design doc §8.3 (``docs/plans/2026-04-28-sparse-autoencoder-design.md``)
and Orlov 2026 §4.2 / Heap et al. 2026: the recommended null model for SAE
interpretability is to train an SAE on activations from a *randomly-initialized*
(untrained) encoder of the same architecture. If the trained-encoder SAE is not
*meaningfully* more interpretable than this null, then a substantial fraction
of recovered features may reflect statistical regularities of the data + the
architecture's induction biases rather than learned computational strategies.

Acceptance criterion (design doc §9.2):
    trained_SAE_interpretable_fraction > 1.5 × random_SAE_interpretable_fraction

This script is the FIRST half of the null pipeline:

1. Load the canonical config (``configs/resdec_mhe/canonical.yaml`` merged on
   top of ``configs/default.yaml``) — exactly the architecture we use, but
   **NO checkpoint loading**. Weights are freshly initialized via
   ``build_model_from_config(model_cfg)``.
2. Set deterministic seed (``torch.manual_seed`` + ``np.random.seed``) BEFORE
   building so weight initialization is reproducible.
3. Forward all 516 subjects through this random encoder once (eval mode, no
   grad), batch size = 64.
4. Capture both ``attended`` ``[B, d_fused=64]`` (post-PathologyStratifiedAttention,
   ``full_model.py:547``) and ``fused`` ``[B, 31, d_fused=64]`` (post-FusionLayer,
   ``full_model.py:534``) — same layers as ``extract_sae_activations.py``.
5. Persist as ``outputs/redesign/sae/random_encoder/activations_{layer}_seed{S}.npz``.

Note: unlike ``extract_sae_activations.py`` (which iterates over 5 fold
checkpoints and concatenates 5 × 516 ≈ 2580 rows), this script forwards all
516 subjects through ONE random encoder. The "fold" axis is meaningless for a
random encoder — there is no per-fold training. The user spec mandates a
single random encoder ("a randomly-initialized encoder, same architecture,
untrained weights, same N=516 inputs"), so we use a single fold's dataloader
to enumerate all subjects exactly once.

Usage
-----
    PYTHONPATH=<worktree-root> \\
    CUDA_VISIBLE_DEVICES=1 \\
    uv run python scripts/resdec_mhe/interpretability/extract_sae_activations_random.py \\
        --config configs/resdec_mhe/canonical.yaml \\
        --out-dir outputs/redesign/sae/random_encoder \\
        --layers attended fused \\
        --seed 0

Arguments
---------
    --config <path>      Phase YAML merged on top of configs/default.yaml
                         (default: ``configs/resdec_mhe/canonical.yaml``).
    --splits-path <path> Splits JSON (default: ``outputs/splits.json``).
    --out-dir <path>     Output directory (created if missing).
    --layers <list>      Which layers to extract; choices ``attended``, ``fused``.
    --device             Torch device (default: ``cuda``).
    --batch-size         Forward-pass batch size (default: 64).
    --seed <int>         Reproducibility seed for random weight init (default: 0).
    --fold <int>         Which fold's split to use to enumerate the 516 subjects
                         (default: 0; choice does not affect activations since
                         we go through both train + val).
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch

_WORKTREE_ROOT = Path(__file__).resolve().parents[3]
if not (_WORKTREE_ROOT / "src").is_dir():
    raise RuntimeError(
        f"sys.path bootstrap failed: {_WORKTREE_ROOT}/src not found; "
        "set PYTHONPATH=<worktree-root>."
    )
sys.path.insert(0, str(_WORKTREE_ROOT))

logger = logging.getLogger(__name__)


def _enumerate_all_subjects(loaders):
    """Yield batches from every loader (train + val) so all 516 subjects are covered once."""
    for loader in loaders:
        if loader is None:
            continue
        for batch in loader:
            yield batch


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "--config",
        default="configs/resdec_mhe/canonical.yaml",
        help="Phase YAML merged on top of configs/default.yaml.",
    )
    p.add_argument(
        "--splits-path",
        default="outputs/splits.json",
        help="Splits JSON path.",
    )
    p.add_argument(
        "--out-dir",
        default="outputs/redesign/sae/random_encoder",
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
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Deterministic seed for random weight init.",
    )
    p.add_argument(
        "--fold",
        type=int,
        default=0,
        help=(
            "Fold index to use for enumerating all subjects via train + val "
            "loaders. Choice does not affect the activations themselves "
            "since both splits are walked."
        ),
    )
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    # Local imports so the module is light when not invoked as a CLI.
    import pandas as pd
    from omegaconf import OmegaConf

    from src.data.constants import CELL_TYPE_ORDER
    from src.data.datamodule import CognitiveResilienceDataModule
    from src.data.splits import get_fold_subjects, load_splits
    from src.models.full_model import build_model_from_config
    from src.utils.provenance import git_sha

    # ─────────────────────────────────────────────────────────────────────
    # Resolve paths.
    # ─────────────────────────────────────────────────────────────────────
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = _WORKTREE_ROOT / config_path
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = _WORKTREE_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    splits_path = Path(args.splits_path)
    if not splits_path.is_absolute():
        splits_path = _WORKTREE_ROOT / splits_path

    # ─────────────────────────────────────────────────────────────────────
    # Build canonical config and a build-from-config (NO checkpoint!) encoder.
    # ─────────────────────────────────────────────────────────────────────
    cfg = OmegaConf.merge(
        OmegaConf.load(_WORKTREE_ROOT / "configs" / "default.yaml"),
        OmegaConf.load(config_path),
    )
    OmegaConf.set_struct(cfg, False)
    # Match the trained-extract path: deterministic head, no Bayesian guide.
    cfg.model.head.type = "deterministic"
    cfg.data.fold = int(args.fold)

    # Reproducibility — set BEFORE building the model so weight init is
    # deterministic. This is the random null's only "tunable" hyperparam.
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))
    logger.info("Random seed for weight init: %d", int(args.seed))

    # Build encoder from config — fresh random weights, NO load_state_dict.
    logger.info("Building random ResDec-MHE encoder via build_model_from_config")
    model = build_model_from_config(cfg.model)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Random encoder parameters: %d", n_params)

    torch_device = torch.device(
        args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    )
    model = model.to(torch_device).eval()

    # ─────────────────────────────────────────────────────────────────────
    # Build dataloader — same as the trained-extract pipeline so subjects
    # enumerate identically (516 rows total per fold's train + val).
    # ─────────────────────────────────────────────────────────────────────
    splits = load_splits(str(splits_path))
    metadata_path = Path(cfg.data.metadata_path) / "metadata.csv"
    if not metadata_path.is_absolute():
        metadata_path = _WORKTREE_ROOT / metadata_path
    metadata_csv = pd.read_csv(metadata_path)

    fold_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    OmegaConf.set_struct(fold_cfg, False)
    fold_cfg.data.fold = int(args.fold)
    # Override DataLoader batch size for forward pass; safe to be larger than
    # training since this is no-grad.
    fold_cfg.data.dataloader.batch_size = int(args.batch_size)

    dm = CognitiveResilienceDataModule(
        config=fold_cfg, metadata=metadata_csv, splits=splits,
        fold_idx=int(args.fold),
        precomputed_dir=fold_cfg.data.precomputed_dir,
        adata=None,
    )
    dm.setup(stage="fit")

    val_subjects: set[str] = set(
        map(str, get_fold_subjects(splits, fold_idx=int(args.fold), split_type="val"))
    )

    # ─────────────────────────────────────────────────────────────────────
    # Forward all subjects, capture both layers in one pass.
    # ─────────────────────────────────────────────────────────────────────
    layers = list(args.layers)
    cell_types_array = (
        np.array(list(CELL_TYPE_ORDER), dtype=object) if "fused" in layers else None
    )

    per_layer_acts: dict[str, list[np.ndarray]] = {layer: [] for layer in layers}
    sids: list[str] = []
    is_val: list[bool] = []

    loaders = [dm.train_dataloader(), dm.val_dataloader()]

    with torch.no_grad():
        for batch in _enumerate_all_subjects(loaders):
            batch_d = {
                k: (v.to(torch_device) if torch.is_tensor(v) else v)
                for k, v in batch.items()
            }
            batch_sids = list(batch_d.get("subject_ids", []))

            enc_out = model(
                region_pseudobulk=batch_d.get("region_pseudobulk"),
                region_mask=batch_d.get("region_mask"),
                pseudobulk=batch_d.get("pseudobulk"),
                ccc_edge_index=batch_d.get("ccc_edge_index"),
                ccc_edge_type=batch_d.get("ccc_edge_type"),
                ccc_edge_attr=batch_d.get("ccc_edge_attr"),
                cell_data=batch_d.get("cell_data"),
                cell_offsets=batch_d.get("cell_offsets"),
                cell_type_mask=batch_d.get("cell_type_mask"),
                pathology=batch_d.get("pathology"),
                return_embeddings=True,
            )
            embeddings = enc_out["embeddings"]
            for layer in layers:
                if layer not in embeddings:
                    raise KeyError(
                        f"layer {layer!r} not in embeddings; available: "
                        f"{list(embeddings.keys())}"
                    )
                per_layer_acts[layer].append(
                    embeddings[layer].detach().cpu().numpy().astype(np.float32)
                )
            sids.extend(batch_sids)
            is_val.extend(str(s) in val_subjects for s in batch_sids)

    if torch_device.type == "cuda":
        torch.cuda.empty_cache()

    sids_arr = np.array([str(s) for s in sids], dtype=object)
    is_val_arr = np.array(is_val, dtype=bool)
    fold_indices_arr = np.full(len(sids), int(args.fold), dtype=np.int64)
    logger.info("Forward complete: %d subjects (val=%d)", len(sids), int(is_val_arr.sum()))

    # ─────────────────────────────────────────────────────────────────────
    # Persist per-layer .npz at the random_encoder out_dir.
    # ─────────────────────────────────────────────────────────────────────
    git_commit = git_sha(_WORKTREE_ROOT)
    for layer in layers:
        arr = np.concatenate(per_layer_acts[layer], axis=0)
        out_npz = out_dir / f"activations_{layer}_seed{int(args.seed)}.npz"
        np.savez(
            out_npz,
            activations=arr,
            subject_ids=sids_arr,
            fold_indices=fold_indices_arr,
            is_val=is_val_arr,
            **(
                {"cell_types": cell_types_array}
                if cell_types_array is not None and layer == "fused"
                else {}
            ),
            layer=np.array(layer, dtype=object),
            seed=np.array(int(args.seed), dtype=np.int64),
            random_init=np.array(True, dtype=bool),
            git_commit=np.array(git_commit, dtype=object),
        )
        logger.info(
            "wrote %s (shape=%s, seed=%d, random_init=True)",
            out_npz, arr.shape, int(args.seed),
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
