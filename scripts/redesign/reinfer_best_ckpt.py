"""Re-infer best-epoch predictions for a Phase 2 fold.

Rationale: the training loop's `val_predictions_final.npz` captures the FINAL
epoch (epoch 60 on seed-42 runs), which is post-overfit. `ModelCheckpoint` saved
the best-by-val/r2 checkpoint at an earlier epoch — this script loads that
checkpoint, runs `trainer.validate()`, and dumps per-subject predictions to
`val_predictions_best.npz` alongside the existing final-epoch file.

Output layout per fold:
    fold{N}/
      val_predictions_final.npz   (unchanged — final epoch, kept for audit)
      val_predictions_best.npz    (NEW — best-by-val/r2 checkpoint)
      best_summary.json           (NEW — val/r2, val/mae, val/rmse, val/pearson_r,
                                   val/spearman_rho, ckpt_filename)

Uses a `reinfer_tmp/` sibling directory as `default_root_dir` so the existing
`val_predictions_final.npz` isn't clobbered; `reinfer_tmp/` is removed at end.

Usage:
    CUDA_VISIBLE_DEVICES=0 \\
    PYTHONPATH=<worktree-root> \\
    uv run python scripts/redesign/reinfer_best_ckpt.py \\
        --config configs/redesign/p5_phase2_residual.yaml \\
        --fold 0 \\
        --output-dir outputs/redesign/p5_phase2_residual
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
from pathlib import Path

import lightning.pytorch as pl
import pandas as pd
import torch
from omegaconf import OmegaConf

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
        epoch = int(m.group(1))
        r2 = float(m.group(2))
        if best is None or r2 > best[2]:
            best = (p, epoch, r2)
    if best is None:
        raise FileNotFoundError(
            f"No best-*.ckpt files matching pattern in {ckpt_dir}"
        )
    return best


def main(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO)

    default_cfg = OmegaConf.load("configs/default.yaml")
    override_cfg = OmegaConf.load(args.config)
    cfg = OmegaConf.merge(default_cfg, override_cfg)
    OmegaConf.set_struct(cfg, False)
    cfg.model.head.type = "deterministic"
    cfg.data.fold = int(args.fold)

    pl.seed_everything(int(cfg.experiment.seed), workers=True)
    torch.set_float32_matmul_precision("high")

    fold_dir = Path(args.output_dir) / f"fold{args.fold}"
    ckpt_dir = fold_dir / "checkpoints"
    ckpt_path, best_epoch, ckpt_r2 = _pick_max_r2_ckpt(ckpt_dir)
    logger.info(
        "fold %d: selected %s (epoch %d, filename-R²=%.4f)",
        args.fold, ckpt_path.name, best_epoch, ckpt_r2,
    )

    splits_path = Path(args.splits_path)
    metadata_csv = Path(cfg.data.metadata_path) / "metadata.csv"
    splits = load_splits(str(splits_path))
    metadata = pd.read_csv(metadata_csv)

    precomputed_dir = args.precomputed_dir or cfg.data.get("precomputed_dir", None)
    if precomputed_dir is None:
        raise ValueError(
            "No precomputed_dir. Pass --precomputed-dir or set data.precomputed_dir."
        )

    dm = CognitiveResilienceDataModule(
        config=cfg,
        metadata=metadata,
        splits=splits,
        fold_idx=args.fold,
        precomputed_dir=precomputed_dir,
        adata=None,
    )
    # CognitiveResilienceDataModule.setup only creates _val_ds under stage="fit"
    # or stage is None. trainer.validate() passes stage="validate" which hits
    # neither branch, so _val_ds would stay None and val_dataloader() returns
    # None. Call setup("fit") explicitly to materialize both _train_ds and
    # _val_ds (train is cheap — index-only, no data copy — per the datamodule's
    # own comment at src/data/datamodule.py:99-103).
    dm.setup(stage="fit")

    model = ResDecLightningModule.load_from_checkpoint(
        str(ckpt_path), config=cfg, map_location="cpu"
    )

    tmp_dir = fold_dir / "reinfer_tmp"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        precision=str(cfg.training.get("precision", "bf16-mixed")),
        default_root_dir=str(tmp_dir),
    )
    val_results = trainer.validate(model, datamodule=dm, verbose=True)
    logger.info("fold %d: VAL_RESULTS_BEST=%s", args.fold, val_results)

    tmp_npz = tmp_dir / "val_predictions_final.npz"
    final_best_npz = fold_dir / "val_predictions_best.npz"
    if not tmp_npz.exists():
        raise FileNotFoundError(
            f"Expected prediction dump not found at {tmp_npz}. The Lightning "
            f"module writes it via trainer.log_dir — check default_root_dir."
        )
    if final_best_npz.exists():
        final_best_npz.unlink()
    tmp_npz.rename(final_best_npz)
    shutil.rmtree(tmp_dir)

    summary = {
        "fold": args.fold,
        "config": args.config,
        "ckpt_filename": ckpt_path.name,
        "ckpt_epoch": best_epoch,
        "ckpt_filename_r2": ckpt_r2,
        "val_results": val_results,
        "val_predictions_best_npz": str(final_best_npz),
    }
    summary_path = fold_dir / "best_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    logger.info("fold %d: wrote %s and %s", args.fold, final_best_npz, summary_path)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Re-infer best-epoch predictions.")
    p.add_argument("--config", default="configs/redesign/p5_phase2_residual.yaml")
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--output-dir", default="outputs/redesign/p5_phase2_residual")
    p.add_argument("--splits-path", default="outputs/splits.json")
    p.add_argument("--precomputed-dir", default=None)
    main(p.parse_args())
