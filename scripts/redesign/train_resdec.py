"""ResDec-H3 training entry (Phase 1 baseline: encoder + bare head, no TabPFN residual yet).

Loads configs/default.yaml + configs/redesign/<phase>.yaml (merged), constructs
the existing CognitiveResilienceDataModule + ResDecLightningModule, and runs
Lightning Trainer on a single fold. Saves a per-fold summary JSON with the
validate() results.

Design note: this is intentionally SIMPLER than scripts/training/train.py —
no HPO hooks, no TensorBoard logger, no checkpointing, no ExperimentManager,
no DDP coordination. The goal is to smoke-test the Phase-1 ResDec-H3 head
end-to-end on a single GPU.

Usage
-----
    PYTHONPATH=<worktree-root> \\
    CUDA_VISIBLE_DEVICES=0 \\
    uv run python scripts/redesign/train_resdec.py \\
        --config configs/redesign/p5_phase1_baseline.yaml \\
        --fold 0 \\
        --max-epochs 60
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import lightning.pytorch as pl
import pandas as pd
import torch
from omegaconf import OmegaConf

from src.data.datamodule import CognitiveResilienceDataModule
from src.data.splits import load_splits
from src.training.resdec_lightning_module import ResDecLightningModule

logger = logging.getLogger(__name__)


def main(args: argparse.Namespace) -> None:
    # ------------------------------------------------------------------ #
    # Config loading (default + phase override)                          #
    # ------------------------------------------------------------------ #
    default_cfg = OmegaConf.load("configs/default.yaml")
    override_cfg = OmegaConf.load(args.config)
    cfg = OmegaConf.merge(default_cfg, override_cfg)
    OmegaConf.set_struct(cfg, False)

    # Force the deterministic head path so the encoder's Bayesian SVI
    # machinery is not engaged. ResDec-H3 produces its own scalar readout
    # and ignores the encoder's prediction_head.
    cfg.model.head.type = "deterministic"
    if args.max_epochs is not None:
        cfg.training.max_epochs = args.max_epochs

    torch.manual_seed(int(cfg.experiment.seed))
    torch.set_float32_matmul_precision("high")

    # ------------------------------------------------------------------ #
    # Splits + metadata                                                  #
    # ------------------------------------------------------------------ #
    splits_path = Path(args.splits_path)
    if not splits_path.exists():
        raise FileNotFoundError(f"Splits file not found: {splits_path}")
    splits = load_splits(str(splits_path))
    logger.info("Loaded splits from %s", splits_path)

    metadata_csv = Path(cfg.data.metadata_path) / "metadata.csv"
    if not metadata_csv.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {metadata_csv}")
    metadata = pd.read_csv(metadata_csv)

    precomputed_dir = args.precomputed_dir or cfg.data.get("precomputed_dir", None)
    if precomputed_dir is None:
        raise ValueError(
            "No precomputed_dir available. Pass --precomputed-dir or set "
            "data.precomputed_dir in the config."
        )

    # ------------------------------------------------------------------ #
    # DataModule                                                         #
    # ------------------------------------------------------------------ #
    dm = CognitiveResilienceDataModule(
        config=cfg,
        metadata=metadata,
        splits=splits,
        fold_idx=args.fold,
        precomputed_dir=precomputed_dir,
        adata=None,
    )
    logger.info("Fold %d DataModule created (precomputed=%s)", args.fold, precomputed_dir)

    # ------------------------------------------------------------------ #
    # Lightning module                                                   #
    # ------------------------------------------------------------------ #
    model = ResDecLightningModule(cfg)
    logger.info("ResDecLightningModule built (d_fused=%d)", int(cfg.model.d_fused))

    # ------------------------------------------------------------------ #
    # Trainer — minimal: no logger, no checkpointing                     #
    # ------------------------------------------------------------------ #
    trainer = pl.Trainer(
        max_epochs=int(cfg.training.max_epochs),
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=True,
        precision=32,
        deterministic=False,
    )
    trainer.fit(model, datamodule=dm)

    # Explicit final val pass so summary JSON always contains a metric
    val_results = trainer.validate(model, datamodule=dm, verbose=False)
    print("VAL_RESULTS:", val_results)

    # ------------------------------------------------------------------ #
    # Summary JSON                                                       #
    # ------------------------------------------------------------------ #
    out_dir = Path(args.output_dir) / f"fold{args.fold}"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "fold": args.fold,
        "config": args.config,
        "max_epochs": int(cfg.training.max_epochs),
        "val_results": val_results,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("Wrote summary to %s", out_dir / "summary.json")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser(description="ResDec-H3 Phase-1 training entry")
    p.add_argument("--config", default="configs/redesign/p5_phase1_baseline.yaml",
                   help="Phase override YAML (merged on top of configs/default.yaml).")
    p.add_argument("--fold", type=int, default=0, help="CV fold index (0-indexed).")
    p.add_argument("--max-epochs", type=int, default=None,
                   help="Override cfg.training.max_epochs for smoke runs.")
    p.add_argument("--output-dir", default="outputs/redesign/p5_phase1",
                   help="Root output directory; a fold<N>/ subdir is created for artifacts.")
    p.add_argument("--splits-path", default="outputs/splits.json",
                   help="Path to 5-fold splits JSON.")
    p.add_argument("--precomputed-dir", default=None,
                   help="Override cfg.data.precomputed_dir (optional).")
    main(p.parse_args())
