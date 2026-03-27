#!/usr/bin/env python3
"""
VRAM profiler — runs the ACTUAL train_fn from hpo.py with real data.

No approximations, no separate model building, no shortcuts. This runs
exactly what HPO runs, on the same data, with the same config.

Usage:
    export LD_LIBRARY_PATH=".venv/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib:${LD_LIBRARY_PATH:-}"
    .venv/bin/python scripts/profile_vram.py \
        --config configs/hpo_round6.yaml \
        --precomputed-dir data/precomputed/rosmap \
        --splits-path outputs/splits.json \
        --batch-size 32 --fusion-type cross_attention --fusion-n-heads 8

    # Compare batch sizes:
    .venv/bin/python scripts/profile_vram.py ... --batch-size 24
"""

import argparse
import json
import logging
import os
import subprocess
import time

import torch
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)


def _gb(n: int) -> float:
    return n / 1024**3


def _nvidia_smi_gpu_mb(gpu_idx: int) -> int:
    """Get GPU memory used by this process from nvidia-smi."""
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True,
    )
    pid = os.getpid()
    for line in result.stdout.strip().split("\n"):
        parts = line.split(",")
        if len(parts) == 2 and parts[0].strip() == str(pid):
            return int(parts[1].strip())
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="VRAM profiler using the actual HPO train_fn."
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--precomputed-dir", type=str, required=True)
    parser.add_argument("--splits-path", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--fusion-type", type=str, default=None)
    parser.add_argument("--fusion-n-heads", type=int, default=None)
    parser.add_argument("--max-epochs", type=int, default=5,
                        help="Epochs to run (default: 5, enough for VRAM to stabilize)")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    # Load and override config — same as hpo.py main() does
    from src.utils.config import load_config
    config = load_config(args.config)

    overrides = []
    if args.batch_size:
        overrides.append(f"data.dataloader.batch_size={args.batch_size}")
    if args.fusion_type:
        overrides.append(f"model.fusion.type={args.fusion_type}")
    if args.fusion_n_heads:
        overrides.append(f"model.fusion.n_heads={args.fusion_n_heads}")
    overrides.append(f"training.max_epochs={args.max_epochs}")
    overrides.append(f"data.precomputed_dir={args.precomputed_dir}")

    for override in overrides:
        key, val = override.split("=", 1)
        try:
            val = int(val)
        except ValueError:
            pass
        OmegaConf.update(config, key, val)

    OmegaConf.update(config, "data.splits.n_folds", 1)

    bs = config.data.dataloader.batch_size
    ft = config.model.fusion.type
    fh = config.model.fusion.n_heads

    # Pin GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    torch.set_float32_matmul_precision("high")

    device_props = torch.cuda.get_device_properties(0)
    total_mem = device_props.total_memory
    print(f"GPU {args.gpu}: {device_props.name}, {_gb(total_mem):.1f} GB")
    print(f"Config: bs={bs}, fusion={ft}, heads={fh}, epochs={args.max_epochs}")
    print(f"  precision={config.training.get('precision')}")
    print(f"  gradient_checkpointing={config.model.get('use_gradient_checkpointing')}")

    # Load data — same as hpo.py main() does
    from src.data.splits import load_splits
    import pandas as pd
    from pathlib import Path

    splits = load_splits(args.splits_path)
    metadata_path = Path(config.data.metadata_path)
    metadata = pd.read_csv(metadata_path / "metadata.csv")

    from src.data.datasets import PrecomputedDataset
    from scripts.hpo import _collect_all_subject_ids
    all_subject_ids = _collect_all_subject_ids(splits)
    preloaded_cache = PrecomputedDataset.load_subject_cache(
        args.precomputed_dir, all_subject_ids,
    )

    # Build the ray_config as Ax/Sobol would — use current config values
    ray_config = {
        "lr": float(config.training.optimizer.lr),
        "dropout": float(config.model.dropout),
        "weight_decay": float(config.training.optimizer.weight_decay),
        "beta": float(config.training.loss.beta),
        "guide_lr": float(config.training.optimizer.guide_lr),
        "fusion_type": str(ft),
        "fusion_n_heads": int(fh),
        "tau_min": float(config.training.temperature_annealing.tau_min),
        "anneal_epochs": int(config.training.temperature_annealing.anneal_epochs),
        "gene_gate_temp": float(config.model.gene_gate.initial_temperature),
    }

    base_config = OmegaConf.to_container(config, resolve=True)

    print(f"\nHP config: {ray_config}")
    print(f"\nStarting train_fn (same code path as HPO)...")

    # Mock tune.report and TuneReportCheckpointCallback since we're outside Ray
    from unittest.mock import patch, MagicMock
    import ray.tune
    import ray.tune.integration.pytorch_lightning as ptl_integration

    torch.cuda.reset_peak_memory_stats(0)
    t0 = time.perf_counter()

    # Call the actual train_fn — same function HPO calls
    from scripts.hpo import train_fn
    with patch.object(ray.tune, "report", MagicMock()), \
         patch.object(ptl_integration, "TuneReportCheckpointCallback",
                      lambda **kw: __import__("lightning.pytorch", fromlist=["Callback"]).Callback()):
            train_fn(
            ray_config=ray_config,
            base_config=base_config,
            splits=splits,
            metadata=metadata,
            preloaded_cache=preloaded_cache,
            n_folds=1,
        )

    elapsed = time.perf_counter() - t0

    # Measure
    torch.cuda.synchronize(torch.device("cuda:0"))
    peak_alloc = torch.cuda.max_memory_allocated(0)
    peak_reserved = torch.cuda.max_memory_reserved(0)
    smi_mb = _nvidia_smi_gpu_mb(args.gpu)

    print(f"\n{'='*70}")
    print(f"VRAM RESULTS ({args.max_epochs} epochs, {elapsed:.0f}s)")
    print(f"{'='*70}")
    print(f"  Peak allocated (PyTorch):  {_gb(peak_alloc):6.2f} GB")
    print(f"  Peak reserved (allocator): {_gb(peak_reserved):6.2f} GB")
    print(f"  nvidia-smi (total process):{smi_mb / 1024:6.2f} GB")
    print(f"  GPU total:                 {_gb(total_mem):6.2f} GB")
    print(f"  Headroom (nvidia-smi):     {_gb(total_mem) - smi_mb / 1024:6.2f} GB")
    print(f"{'='*70}")
    print(f"  Config: bs={bs}, fusion={ft}, heads={fh}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    main()
