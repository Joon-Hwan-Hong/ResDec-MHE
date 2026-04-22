"""ResDec-H3 training entry (Phase 1 baseline: encoder + bare head, no TabPFN residual yet).

Loads configs/default.yaml + configs/redesign/<phase>.yaml (merged), constructs
the existing CognitiveResilienceDataModule + ResDecLightningModule, and runs
Lightning Trainer on a single fold. Saves a per-fold summary JSON with the
validate() results.

Design note: this is intentionally SIMPLER than scripts/training/train.py —
no HPO hooks, no TensorBoard logger, no ExperimentManager, no DDP coordination.
A minimal ModelCheckpoint (save_last + best-by-val/r2) IS enabled so runs can be
resumed and best weights recovered. The goal is to smoke-test the Phase-1
ResDec-H3 head end-to-end on a single GPU.

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
from lightning.pytorch.callbacks import ModelCheckpoint
from omegaconf import OmegaConf

from src.data.datamodule import CognitiveResilienceDataModule
from src.data.embedding_datamodule import EmbeddingDataModule
from src.data.splits import load_splits
from src.training.resdec_frozen_lightning_module import ResDecFrozenLightningModule
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

    pl.seed_everything(int(cfg.experiment.seed), workers=True)
    torch.set_float32_matmul_precision("high")

    # ------------------------------------------------------------------ #
    # Splits + metadata                                                  #
    # ------------------------------------------------------------------ #
    splits_path = Path(args.splits_path)
    if not splits_path.exists():
        raise FileNotFoundError(f"Splits file not found: {splits_path}")

    metadata_csv = Path(cfg.data.metadata_path) / "metadata.csv"
    if not metadata_csv.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {metadata_csv}")

    # ------------------------------------------------------------------ #
    # DataModule + LightningModule (branch on cached-embeddings flag)    #
    # ------------------------------------------------------------------ #
    use_cached = bool(cfg.data.get("use_cached_embeddings", False))
    if use_cached:
        # Frozen-encoder path (option 2 of the D-OOM fix). No encoder forward
        # at train time — the ResDec-H3 head is trained on cached `attended`
        # embeddings at full-cohort batch.
        embeddings_npz = Path(
            args.embeddings_npz
            or cfg.data.get("embeddings_npz", "data/redesign/encoder_embeddings.npz")
        )
        if not embeddings_npz.exists():
            raise FileNotFoundError(
                f"Cached embeddings not found: {embeddings_npz}. Run "
                f"scripts/redesign/precompute_encoder_embeddings.py first."
            )
        dl_cfg = cfg.data.get("dataloader", {}) or {}
        batch_size = int(dl_cfg.get("batch_size", 500))
        dm = EmbeddingDataModule(
            embeddings_npz=embeddings_npz,
            splits_path=splits_path,
            meta_csv=metadata_csv,
            fold=args.fold,
            batch_size=batch_size,
        )
        model = ResDecFrozenLightningModule(cfg)
        logger.info(
            "Frozen-encoder path: fold=%d, embeddings=%s, batch_size=%d, d_fused=%d",
            args.fold, embeddings_npz, batch_size, int(cfg.model.d_fused),
        )
    else:
        # Live-encoder path (original): encoder runs per step, head trains on
        # its `attended` output.
        splits = load_splits(str(splits_path))
        logger.info("Loaded splits from %s", splits_path)
        metadata = pd.read_csv(metadata_csv)

        precomputed_dir = args.precomputed_dir or cfg.data.get("precomputed_dir", None)
        if precomputed_dir is None:
            raise ValueError(
                "No precomputed_dir available. Pass --precomputed-dir or set "
                "data.precomputed_dir in the config."
            )

        dm = CognitiveResilienceDataModule(
            config=cfg,
            metadata=metadata,
            splits=splits,
            fold_idx=args.fold,
            precomputed_dir=precomputed_dir,
            adata=None,
        )
        logger.info(
            "Fold %d DataModule created (precomputed=%s)", args.fold, precomputed_dir,
        )
        model = ResDecLightningModule(cfg)
        logger.info("ResDecLightningModule built (d_fused=%d)", int(cfg.model.d_fused))

    # ------------------------------------------------------------------ #
    # Output dir (computed early so checkpoints can be written into it)  #
    # ------------------------------------------------------------------ #
    out_dir = Path(args.output_dir) / f"fold{args.fold}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Trainer — minimal checkpointing (last + best-by-val/r2)             #
    # ------------------------------------------------------------------ #
    checkpoint_cb = ModelCheckpoint(
        dirpath=str(out_dir / "checkpoints"),
        save_last=True,
        save_top_k=1,
        monitor="val/r2",
        mode="max",
        filename="best-{epoch}-{val/r2:.4f}",
        auto_insert_metric_name=False,
    )

    trainer = pl.Trainer(
        max_epochs=int(cfg.training.max_epochs),
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=False,
        enable_checkpointing=True,
        callbacks=[checkpoint_cb],
        enable_progress_bar=True,
        # Full-cohort NPT (bs~412) makes fp32 OOM on a 48 GB GPU; bf16-mixed
        # (Ada-friendly) halves activation memory. See default.yaml precision
        # notes — results are tied to precision setting but this script is a
        # Phase-1 smoke run, not a reproducibility-critical production run.
        precision="bf16-mixed",
        enable_model_summary=True,
    )
    trainer.fit(model, datamodule=dm)

    # Explicit final val pass so summary JSON always contains a metric
    val_results = trainer.validate(model, datamodule=dm, verbose=False)
    print("VAL_RESULTS:", val_results)

    # ------------------------------------------------------------------ #
    # Summary JSON                                                       #
    # ------------------------------------------------------------------ #
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
    p.add_argument("--embeddings-npz", default=None,
                   help="Override cfg.data.embeddings_npz (optional, frozen-encoder path only).")
    main(p.parse_args())
