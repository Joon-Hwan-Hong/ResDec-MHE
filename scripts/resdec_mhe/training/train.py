"""ResDec-MHE training entry.

Loads configs/default.yaml + configs/resdec_mhe/<phase>.yaml (merged), constructs
the existing CognitiveResilienceDataModule + ResDecLightningModule, and runs
Lightning Trainer on a single fold. Saves a per-fold summary JSON with the
validate() results.

Design note: this is intentionally SIMPLER than scripts/training/train.py —
no HPO hooks, no TensorBoard logger, no ExperimentManager, no DDP coordination.
A minimal ModelCheckpoint (save_last + best-by-val/r2) IS enabled so runs can be
resumed and best weights recovered. The goal is to smoke-test the
ResDec-MHE head end-to-end on a single GPU.

Usage
-----
    PYTHONPATH=<worktree-root> \\
    CUDA_VISIBLE_DEVICES=0 \\
    uv run python scripts/resdec_mhe/training/train.py \\
        --config configs/resdec_mhe/canonical.yaml \\
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
from lightning.pytorch.loggers import CSVLogger
from omegaconf import OmegaConf

from src.data.datamodule import CognitiveResilienceDataModule
from src.data.embedding_datamodule import EmbeddingDataModule
from src.data.splits import load_splits
from src.training.callbacks import MinEpochEarlyStopping
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
    # machinery is not engaged. ResDec-MHE produces its own scalar readout
    # and ignores the encoder's prediction_head.
    cfg.model.head.type = "deterministic"
    if args.max_epochs is not None:
        cfg.training.max_epochs = args.max_epochs
    if args.seed is not None:
        cfg.experiment.seed = int(args.seed)

    # Propagate fold index into cfg.data so ResDecLightningModule can load the
    # fold-specific TabPFN residual caches (harmless no-op for other paths —
    # Lightning module reads it only when tabpfn_oof_dir is set).
    cfg.data.fold = int(args.fold)

    # Optional CLI overrides for permutation tests that need to swap in
    # shuffled-labels metadata + caches without editing the config file.
    if args.metadata_path is not None:
        cfg.data.metadata_path = args.metadata_path
    if args.tabpfn_oof_dir is not None:
        cfg.data.tabpfn_oof_dir = args.tabpfn_oof_dir
    if args.tabpfn_outer_dir is not None:
        cfg.data.tabpfn_outer_dir = args.tabpfn_outer_dir

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
        # at train time — the ResDec-MHE head is trained on cached `attended`
        # embeddings at full-cohort batch.
        embeddings_npz = Path(
            args.embeddings_npz
            or cfg.data.get("embeddings_npz", "data/redesign/encoder_embeddings.npz")
        )
        if not embeddings_npz.exists():
            raise FileNotFoundError(
                f"Cached embeddings not found: {embeddings_npz}. Run "
                f"scripts/resdec_mhe/precompute_encoder_embeddings.py first."
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

    # Early stopping: active iff cfg.training.early_stopping is present. The
    # existing resdec_mhe runs showed every fold overfits past epoch ~8 on the
    # residual target (seed 42), so the callback is essential for q2+ runs.
    # MinEpochEarlyStopping guards warmup; min_epochs below ES patience prevents
    # premature stop on fold-2-like early peaks.
    callbacks: list = [checkpoint_cb]
    es_cfg = cfg.training.get("early_stopping", None)
    if es_cfg is not None:
        es_cb = MinEpochEarlyStopping(
            min_epochs=int(es_cfg.get("min_epochs", 3)),
            monitor=str(es_cfg.get("monitor", "val/r2")),
            mode=str(es_cfg.get("mode", "max")),
            patience=int(es_cfg.get("patience", 5)),
            min_delta=float(es_cfg.get("min_delta", 0.0)),
            verbose=True,
        )
        callbacks.append(es_cb)
        logger.info("EarlyStopping: %r", es_cb)
    else:
        logger.info("EarlyStopping disabled (no cfg.training.early_stopping block).")

    # Optional CSVLogger (phase 3 task 3.2 diagnostic re-runs). Default=False
    # to preserve the pre-phase3 behaviour (no per-epoch CSV dumps).
    trainer_logger = False
    if args.csv_log:
        trainer_logger = CSVLogger(
            save_dir=str(out_dir), name="csv_logs", version="",
        )

    trainer = pl.Trainer(
        max_epochs=int(cfg.training.max_epochs),
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=trainer_logger,
        enable_checkpointing=True,
        callbacks=callbacks,
        enable_progress_bar=True,
        # Pulls from cfg.training.precision (default.yaml: "bf16-mixed"). Full-
        # cohort NPT (bs~412) makes fp32 OOM on a 48 GB GPU; bf16-mixed (Ada-
        # friendly) halves activation memory. Override via phase YAML for a
        # strict fp32 reproducibility pass.
        precision=str(cfg.training.get("precision", "bf16-mixed")),
        enable_model_summary=True,
        # default_root_dir makes trainer.log_dir resolve to the fold's output
        # directory so per-subject predictions dumped by Option B in
        # ResDecLightningModule land in outputs/<run_root>/.../fold{N}/
        # rather than being overwritten per-fold in cwd.
        default_root_dir=str(out_dir),
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
    p = argparse.ArgumentParser(description="ResDec-MHE training entry")
    p.add_argument("--config", default="configs/resdec_mhe/canonical.yaml",
                   help="Phase override YAML (merged on top of configs/default.yaml).")
    p.add_argument("--fold", type=int, default=0, help="CV fold index (0-indexed).")
    p.add_argument("--max-epochs", type=int, default=None,
                   help="Override cfg.training.max_epochs for smoke runs.")
    p.add_argument("--seed", type=int, default=None,
                   help="Override cfg.experiment.seed (e.g. 43 for sanity rerun).")
    p.add_argument("--output-dir", default="outputs/redesign/p5_phase1",
                   help="Root output directory; a fold<N>/ subdir is created for artifacts.")
    p.add_argument("--splits-path", default="outputs/splits.json",
                   help="Path to 5-fold splits JSON.")
    p.add_argument("--precomputed-dir", default=None,
                   help="Override cfg.data.precomputed_dir (optional).")
    p.add_argument("--embeddings-npz", default=None,
                   help="Override cfg.data.embeddings_npz (optional, frozen-encoder path only).")
    p.add_argument("--csv-log", action="store_true",
                   help="Attach a CSVLogger to capture per-epoch train/val scalars "
                        "(phase-3 task-3.2 diagnostic; default off).")
    p.add_argument("--metadata-path", default=None,
                   help="Override cfg.data.metadata_path (the directory holding "
                        "metadata.csv). Use for permutation tests that need a "
                        "shuffled-labels metadata copy.")
    p.add_argument("--tabpfn-oof-dir", default=None,
                   help="Override cfg.data.tabpfn_oof_dir (per-fold OOF cache).")
    p.add_argument("--tabpfn-outer-dir", default=None,
                   help="Override cfg.data.tabpfn_outer_dir (per-fold outer cache).")
    main(p.parse_args())
