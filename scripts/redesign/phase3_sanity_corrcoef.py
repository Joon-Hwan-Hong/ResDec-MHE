"""Phase 3 Task 3.2 sanity: stage-wise corrcoef on val set.

Loads the best-by-val/r2 Phase-3 ResDec-H3 checkpoint for a single fold,
runs a forward pass over the validation set, gathers per-subject
``stage_1``, ``stage_2``, ``stage_3`` scalars, and prints
``corrcoef(f̂_1, f̂_2)`` and ``corrcoef(f̂_1, f̂_3)``.

Plan target: both correlations must be < 0.3 (stages learning distinct signal).

Usage
-----
    CUDA_VISIBLE_DEVICES=0 \\
    PYTHONPATH=<worktree-root> \\
    uv run python scripts/redesign/phase3_sanity_corrcoef.py \\
        --config configs/redesign/p5_phase2_residual.yaml \\
        --fold 0 \\
        --output-dir outputs/redesign/p5_phase3
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

# Make the script standalone-runnable: ensure the worktree root is on sys.path
# so `src.*` imports resolve without the caller having to set PYTHONPATH.
# Mirrors the pattern used by scripts/redesign/run_tabpfn_attribution.py.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.datamodule import CognitiveResilienceDataModule
from src.data.splits import load_splits
from src.training.resdec_lightning_module import ResDecLightningModule

logger = logging.getLogger(__name__)
_BEST_CKPT_RE = re.compile(r"^best-(\d+)-(\d+\.\d+)\.ckpt$")


def _pick_max_r2_ckpt(ckpt_dir: Path) -> tuple[Path, int, float]:
    best: tuple[Path, int, float] | None = None
    for p in ckpt_dir.glob("best-*.ckpt"):
        m = _BEST_CKPT_RE.match(p.name)
        if not m:
            continue
        epoch, r2 = int(m.group(1)), float(m.group(2))
        if best is None or r2 > best[2]:
            best = (p, epoch, r2)
    if best is None:
        raise FileNotFoundError(f"No best-*.ckpt files in {ckpt_dir}")
    return best


def main(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO)
    default_cfg = OmegaConf.load("configs/default.yaml")
    cfg = OmegaConf.merge(default_cfg, OmegaConf.load(args.config))
    OmegaConf.set_struct(cfg, False)
    cfg.model.head.type = "deterministic"
    cfg.data.fold = int(args.fold)

    fold_dir = Path(args.output_dir) / f"fold{args.fold}"
    ckpt_path, epoch, ckpt_r2 = _pick_max_r2_ckpt(fold_dir / "checkpoints")
    logger.info("Loading %s (epoch=%d, filename-R²=%.4f)", ckpt_path.name, epoch, ckpt_r2)

    splits = load_splits(str(args.splits_path))
    metadata = pd.read_csv(Path(cfg.data.metadata_path) / "metadata.csv")
    dm = CognitiveResilienceDataModule(
        config=cfg, metadata=metadata, splits=splits,
        fold_idx=args.fold,
        precomputed_dir=args.precomputed_dir or cfg.data.precomputed_dir,
        adata=None,
    )
    dm.setup(stage="fit")

    model = ResDecLightningModule.load_from_checkpoint(
        str(ckpt_path), config=cfg, map_location="cpu",
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    stage_1_all: list[torch.Tensor] = []
    stage_2_all: list[torch.Tensor] = []
    stage_3_all: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in dm.val_dataloader():
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            out = model.forward(batch)
            stage_1_all.append(out["stage_1"].detach().cpu())
            stage_2_all.append(out["stage_2"].detach().cpu())
            stage_3_all.append(out["stage_3"].detach().cpu())

    f1 = torch.cat(stage_1_all).float().numpy()
    f2 = torch.cat(stage_2_all).float().numpy()
    f3 = torch.cat(stage_3_all).float().numpy()
    n = len(f1)
    r12 = float(np.corrcoef(f1, f2)[0, 1])
    r13 = float(np.corrcoef(f1, f3)[0, 1])
    r23 = float(np.corrcoef(f2, f3)[0, 1])
    logger.info("n_val=%d, corrcoef(f1,f2)=%.4f, corrcoef(f1,f3)=%.4f, corrcoef(f2,f3)=%.4f",
                n, r12, r13, r23)
    pass_12 = r12 < 0.3
    pass_13 = r13 < 0.3
    print(json.dumps({
        "fold": args.fold,
        "ckpt": ckpt_path.name,
        "n_val": n,
        "corrcoef_f1_f2": r12,
        "corrcoef_f1_f3": r13,
        "corrcoef_f2_f3": r23,
        "stage_1_mean": float(f1.mean()), "stage_1_std": float(f1.std()),
        "stage_2_mean": float(f2.mean()), "stage_2_std": float(f2.std()),
        "stage_3_mean": float(f3.mean()), "stage_3_std": float(f3.std()),
        "pass_f1_f2_lt_0.3": pass_12,
        "pass_f1_f3_lt_0.3": pass_13,
    }, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Phase 3 stage-wise corrcoef sanity check.")
    p.add_argument("--config", default="configs/redesign/p5_phase2_residual.yaml")
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--output-dir", default="outputs/redesign/p5_phase3")
    p.add_argument("--splits-path", default="outputs/splits.json")
    p.add_argument("--precomputed-dir", default=None)
    main(p.parse_args())
