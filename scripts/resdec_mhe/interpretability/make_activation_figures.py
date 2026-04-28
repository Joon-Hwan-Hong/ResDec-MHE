"""Orchestrator: render per-stage activation cascade figure.

Loads fold-0 canonical checkpoint via ``ResDecLightningModule.load_from_checkpoint``,
registers forward hooks on the encoder's top-level named children (plus the
composite head output), runs one val-fold batch, computes per-subject L2
norms of each captured activation, and renders
``fig_per_stage_activation_cascade.{png,pdf}`` via
``plot_per_stage_activation_cascade`` from
``src.visualization.activation_plots``.

CPU-only by default (fold-0 inference on ~100 val subjects at batch=1 is
fast enough that CPU is preferred to avoid GPU conflicts with concurrent
jobs). Use ``--device cuda:0`` to override.
"""
from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

_WORKTREE_ROOT = Path(__file__).resolve().parents[3]
if str(_WORKTREE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKTREE_ROOT))

from src.data.datamodule import CognitiveResilienceDataModule
from src.data.splits import load_splits
from src.training.resdec_lightning_module import ResDecLightningModule
from src.visualization.activation_plots import (
    plot_per_stage_activation_cascade,
)
from src.visualization.theme import apply_theme

logger = logging.getLogger(__name__)
_BEST_CKPT_RE = re.compile(r"^best-(\d+)-(\d+\.\d+)\.ckpt$")


def _pick_max_r2_ckpt(ckpt_dir: Path) -> Path:
    best: tuple[Path, float] | None = None
    for p in ckpt_dir.glob("best-*.ckpt"):
        m = _BEST_CKPT_RE.match(p.name)
        if not m:
            continue
        r2 = float(m.group(2))
        if best is None or r2 > best[1]:
            best = (p, r2)
    if best is None:
        raise FileNotFoundError(f"No best-*.ckpt in {ckpt_dir}")
    return best[0]


def _first_tensor(obj):
    """Recursively extract the first torch.Tensor from obj (dict / tuple / tensor)."""
    if isinstance(obj, torch.Tensor):
        return obj
    if isinstance(obj, dict):
        for v in obj.values():
            t = _first_tensor(v)
            if t is not None:
                return t
    if isinstance(obj, (tuple, list)):
        for v in obj:
            t = _first_tensor(v)
            if t is not None:
                return t
    return None


def _per_subject_norm(tensor: torch.Tensor, batch_size: int) -> np.ndarray:
    """L2 norm per batch element; if tensor lacks a batch dim, repeat for all."""
    x = tensor.detach().float().cpu()
    if x.ndim == 0:
        return np.full(batch_size, float(x), dtype=np.float64)
    if x.size(0) != batch_size:
        # Collapse everything to a single scalar norm; repeat for all subjects.
        return np.full(batch_size, float(x.norm()), dtype=np.float64)
    flat = x.reshape(x.size(0), -1)
    return flat.norm(dim=1).numpy().astype(np.float64)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--config", default="configs/resdec_mhe/canonical.yaml")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument(
        "--canonical-dir", default="outputs/canonical/p5_canonical_seed42",
    )
    p.add_argument("--splits-path", default="outputs/splits.json")
    p.add_argument("--device", default="cpu",
                   help="'cpu' (default) or 'cuda:0' etc.")
    p.add_argument(
        "--out-dir",
        default="outputs/canonical/interpretability/figures/activation",
    )
    p.add_argument(
        "--max-val-subjects", type=int, default=48,
        help="Cap val subjects for single-pass activation capture.",
    )
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    apply_theme()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    cfg = OmegaConf.merge(
        OmegaConf.load("configs/default.yaml"),
        OmegaConf.load(args.config),
    )
    OmegaConf.set_struct(cfg, False)
    cfg.model.head.type = "deterministic"
    cfg.data.fold = int(args.fold)

    fold_dir = Path(args.canonical_dir) / f"fold{args.fold}"
    ckpt_path = _pick_max_r2_ckpt(fold_dir / "checkpoints")
    logger.info("fold %d: loading %s", args.fold, ckpt_path.name)

    splits = load_splits(str(args.splits_path))
    metadata = pd.read_csv(Path(cfg.data.metadata_path) / "metadata.csv")
    dm = CognitiveResilienceDataModule(
        config=cfg, metadata=metadata, splits=splits,
        fold_idx=args.fold,
        precomputed_dir=cfg.data.precomputed_dir,
        adata=None,
    )
    dm.setup(stage="fit")

    model = ResDecLightningModule.load_from_checkpoint(
        str(ckpt_path), config=cfg, map_location="cpu",
    ).to(device).eval()

    # Register hooks on the encoder's top-level named children + head.
    stage_outputs: dict[str, torch.Tensor] = {}
    hook_handles = []

    def _make_hook(name: str):
        def hook(_module, _inputs, output):
            t = _first_tensor(output)
            if t is not None:
                stage_outputs[name] = t

        return hook

    encoder = getattr(model, "encoder", model)
    for name, submod in encoder.named_children():
        hook_handles.append(submod.register_forward_hook(_make_hook(name)))
    head = getattr(model, "head", None)
    if head is not None:
        hook_handles.append(head.register_forward_hook(_make_hook("head")))

    logger.info(
        "registered hooks on %d encoder children + %s head",
        len(list(encoder.named_children())),
        "1" if head is not None else "0",
    )

    # Run forward on ≤ max_val_subjects subjects.
    all_norms: dict[str, list[np.ndarray]] = {}
    seen_subjects = 0
    with torch.no_grad():
        for batch in dm.val_dataloader():
            if seen_subjects >= args.max_val_subjects:
                break
            stage_outputs.clear()
            batch_d = {
                k: v.to(device) if torch.is_tensor(v) else v
                for k, v in batch.items()
            }
            _ = model.forward(batch_d)
            B = next(
                (v.size(0) for v in batch_d.values() if torch.is_tensor(v)),
                1,
            )
            for name, t in stage_outputs.items():
                all_norms.setdefault(name, []).append(
                    _per_subject_norm(t, B),
                )
            seen_subjects += B
    for h in hook_handles:
        h.remove()
    logger.info("collected activations over %d subjects; %d stages",
                seen_subjects, len(all_norms))

    if len(all_norms) < 2:
        logger.error(
            "need ≥2 stages for cascade; captured %d", len(all_norms),
        )
        return 1

    stage_norms_final = {
        name: np.concatenate(chunks) for name, chunks in all_norms.items()
    }

    fig = plot_per_stage_activation_cascade(
        stage_norms_final,
        save_path=out_dir / "fig_per_stage_activation_cascade",
    )
    plt.close(fig)
    logger.info(
        "rendered fig_per_stage_activation_cascade with stages=%s",
        list(stage_norms_final.keys()),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
