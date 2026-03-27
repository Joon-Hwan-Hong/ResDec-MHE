"""Profile training steps with CUDA event timing.

Runs a short training loop and measures wall-clock time for each phase
(data loading, forward, backward, optimizer step) using CUDA events for
GPU-accurate timing. Also reports GPU memory usage.

Unlike torch.profiler, this approach has zero overhead and no risk of OOM
from profiler event storage (Pyro SVI generates millions of ops per step,
which overwhelms torch.profiler's in-memory event buffer even on 250+ GB
RAM machines).

For deeper kernel-level profiling, use NVIDIA Nsight Systems:
    nsys profile -o report .venv/bin/python scripts/profile_training.py ...

Usage:
    .venv/bin/python scripts/profile_training.py \
        --config configs/default.yaml \
        --splits-path outputs/splits.json \
        --precomputed-dir data/precomputed/rosmap/

    # More steps for stable averages:
    .venv/bin/python scripts/profile_training.py \
        --config configs/default.yaml \
        --splits-path outputs/splits.json \
        --precomputed-dir data/precomputed/rosmap/ \
        --n-steps 10 --warmup-steps 3
"""

import argparse
import logging
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch

import pyro
import pyro.poutine
from pyro.infer import TraceMeanField_ELBO
from pyro.infer.autoguide import AutoDiagonalNormal

from src.data.datamodule import CognitiveResilienceDataModule
from src.data.splits import load_splits
from src.models.full_model import build_model_from_config
from src.utils.config import load_config, validate_config
from src.utils.reproducibility import set_seed

logger = logging.getLogger(__name__)


def _move_batch_to_device(
    batch: dict, device: torch.device
) -> dict:
    """Move batch tensors to device (mirrors Lightning's transfer_batch_to_device)."""
    moved = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            moved[k] = v.to(device, non_blocking=True)
        elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], dict):
            moved[k] = [
                {
                    ek: ev.to(device, non_blocking=True)
                    if isinstance(ev, torch.Tensor)
                    else ev
                    for ek, ev in d.items()
                }
                for d in v
            ]
        else:
            moved[k] = v
    return moved


def _forward_kwargs(batch: dict) -> dict:
    """Extract model forward kwargs from batch dict."""
    return dict(
        region_pseudobulk=batch.get("region_pseudobulk"),
        region_mask=batch.get("region_mask"),
        pseudobulk=batch.get("pseudobulk"),
        ccc_edge_index=batch.get("ccc_edge_index"),
        ccc_edge_type=batch.get("ccc_edge_type"),
        ccc_edge_attr=batch.get("ccc_edge_attr"),
        ccc_edge_counts=batch.get("ccc_edge_counts"),
        cells=batch.get("cells"),
        cell_mask=batch.get("cell_mask"),
        cell_type_mask=batch.get("cell_type_mask"),
        pathology=batch.get("pathology"),
        cognition=batch.get("cognition"),
    )


class CUDATimer:
    """Precise GPU timing using CUDA events."""

    def __init__(self, device: torch.device):
        self.device = device
        self.use_cuda = device.type == "cuda"
        self.timings: dict[str, list[float]] = defaultdict(list)
        self._start_event = None
        self._start_wall = None

    def start(self):
        if self.use_cuda:
            self._start_event = torch.cuda.Event(enable_timing=True)
            self._start_event.record()
        self._start_wall = time.perf_counter()

    def stop(self, label: str):
        if self.use_cuda:
            end_event = torch.cuda.Event(enable_timing=True)
            end_event.record()
            torch.cuda.synchronize()
            elapsed_ms = self._start_event.elapsed_time(end_event)
        else:
            elapsed_ms = (time.perf_counter() - self._start_wall) * 1000.0
        self.timings[label].append(elapsed_ms)
        return elapsed_ms

    def summary(self, skip_first: int = 0) -> str:
        """Format timing summary, optionally skipping warmup steps."""
        lines = []
        lines.append(f"\n{'='*80}")
        lines.append("TIMING SUMMARY (milliseconds)")
        if skip_first > 0:
            lines.append(f"  (first {skip_first} steps excluded as warmup)")
        lines.append(f"{'='*80}")
        lines.append(
            f"{'Phase':<25} {'Mean':>10} {'Std':>10} {'Min':>10} "
            f"{'Max':>10} {'Count':>7}"
        )
        lines.append("-" * 80)

        total_mean = 0.0
        for label, times in self.timings.items():
            t = times[skip_first:] if skip_first < len(times) else times
            if not t:
                continue
            import numpy as np
            arr = np.array(t)
            mean = arr.mean()
            std = arr.std()
            if label != "total_step":
                total_mean += mean
            lines.append(
                f"{label:<25} {mean:>10.1f} {std:>10.1f} {arr.min():>10.1f} "
                f"{arr.max():>10.1f} {len(t):>7}"
            )

        # Add total from individual phases
        lines.append("-" * 80)
        if "total_step" in self.timings:
            t = self.timings["total_step"]
            t = t[skip_first:] if skip_first < len(t) else t
            if t:
                import numpy as np
                arr = np.array(t)
                lines.append(
                    f"{'total_step (measured)':<25} {arr.mean():>10.1f} "
                    f"{arr.std():>10.1f} {arr.min():>10.1f} "
                    f"{arr.max():>10.1f} {len(t):>7}"
                )
        lines.append(f"{'total (sum of phases)':<25} {total_mean:>10.1f}")
        lines.append("=" * 80)
        return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Profile training steps with CUDA event timing"
    )
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to config YAML file",
    )
    parser.add_argument(
        "--splits-path", type=str, required=True,
        help="Path to pre-computed splits JSON file",
    )
    parser.add_argument(
        "--precomputed-dir", type=str, required=True,
        help="Path to precomputed feature directory",
    )
    parser.add_argument(
        "--output-dir", type=str, default="outputs/profiling/",
        help="Directory for profiler output files",
    )
    parser.add_argument(
        "--n-steps", type=int, default=6,
        help="Total training steps to run",
    )
    parser.add_argument(
        "--warmup-steps", type=int, default=2,
        help="Steps to exclude from timing averages (JIT/cache warmup)",
    )
    parser.add_argument(
        "--num-workers", type=int, default=0,
        help="DataLoader workers (0 = in-process, avoids shared memory issues)",
    )
    parser.add_argument(
        "--profile-subset", type=str, default=None,
        help="Path to profile_subset.json for reproducible subject selection",
    )
    parser.add_argument(
        "overrides", nargs="*",
        help="Config overrides in dotlist format (e.g., data.dataloader.batch_size=8)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Config ──────────────────────────────────────────────────────────────
    config = load_config(args.config, overrides=args.overrides)
    validate_config(
        config, required_keys=["experiment", "data", "model", "training", "paths"]
    )

    seed = config.experiment.get("seed", 42)
    set_seed(seed, deterministic=False, benchmark=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    precision = config.training.get("precision", "bf16-mixed")
    use_amp = precision in ("bf16-mixed", "16-mixed") and device.type == "cuda"
    amp_dtype = torch.bfloat16 if precision == "bf16-mixed" else torch.float16

    logger.info("Device: %s | Precision: %s | AMP: %s", device, precision, use_amp)

    # ── Model ───────────────────────────────────────────────────────────────
    is_bayesian = config.model.head.type == "bayesian"
    model = build_model_from_config(config.model).to(device)
    model.train()

    guide = None
    elbo = None
    if is_bayesian:
        pyro.clear_param_store()
        guide = AutoDiagonalNormal(model)
        pyro.enable_validation(False)
        elbo = TraceMeanField_ELBO()
        _prototype_guide(model, guide, elbo, config, device)

    # ── Optimizer ───────────────────────────────────────────────────────────
    opt_cfg = config.training.optimizer
    if is_bayesian:
        all_params = list(model.parameters()) + list(guide.parameters())
        optimizer = torch.optim.Adam(
            all_params,
            lr=opt_cfg.lr,
            weight_decay=opt_cfg.get("weight_decay", 0),
            betas=tuple(opt_cfg.get("betas", [0.9, 0.999])),
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=opt_cfg.lr,
            weight_decay=opt_cfg.weight_decay,
            betas=tuple(opt_cfg.get("betas", [0.9, 0.999])),
        )

    clip_val = config.training.get("gradient_clip_val", None)
    params_for_clip = list(model.parameters())
    if guide is not None:
        params_for_clip += list(guide.parameters())

    # ── Data ────────────────────────────────────────────────────────────────
    metadata_path = Path(config.data.metadata_path) / "metadata.csv"
    metadata = pd.read_csv(metadata_path)
    splits = load_splits(args.splits_path)

    dm = CognitiveResilienceDataModule(
        config=config,
        metadata=metadata,
        splits=splits,
        fold_idx=0,
        precomputed_dir=args.precomputed_dir,
    )
    dm.setup(stage="fit")

    from src.data.collate import create_dataloader

    train_ds = dm._train_ds

    # Apply profiling subset if provided
    if args.profile_subset:
        from scripts.profiling_subset import load_profiling_subset
        subset_ids = set(load_profiling_subset(Path(args.profile_subset)))
        original_n = len(train_ds.subject_ids)
        train_ds.subject_ids = [s for s in train_ds.subject_ids if s in subset_ids]
        logger.info(
            "Profile subset: %d -> %d subjects (from %s)",
            original_n, len(train_ds.subject_ids), args.profile_subset,
        )

    dl_kwargs = dict(
        batch_size=config.data.dataloader.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=config.data.dataloader.get("pin_memory", True) and args.num_workers > 0,
        multiregion=True,
        use_hgt_format=True,
    )
    if args.num_workers > 0:
        dl_kwargs["prefetch_factor"] = config.data.dataloader.get("prefetch_factor", 2)
    train_dl = create_dataloader(train_ds, **dl_kwargs)

    # ── GPU memory baseline ─────────────────────────────────────────────────
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        mem_after_setup = torch.cuda.memory_allocated() / 1e9
        logger.info("GPU memory after setup: %.2f GB", mem_after_setup)

    # ── Training loop with timing ────────────────────────────────────────────
    timer = CUDATimer(device)
    total_steps = args.n_steps
    logger.info(
        "Running %d steps (%d warmup + %d measured)",
        total_steps, args.warmup_steps, total_steps - args.warmup_steps,
    )

    batch_iter = iter(train_dl)
    edge_counts_per_step: list[int] = []
    for step_idx in range(total_steps):
        # ── Data loading ──
        timer.start()
        try:
            batch = next(batch_iter)
        except StopIteration:
            batch_iter = iter(train_dl)
            batch = next(batch_iter)
        batch = _move_batch_to_device(batch, device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        timer.stop("data_load+transfer")

        # Track max edge count per batch (explains step time variance)
        if "ccc_edge_counts" in batch:
            edge_counts_per_step.append(batch["ccc_edge_counts"].max().item())

        # ── Forward ──
        timer.start()
        if is_bayesian:
            with torch.amp.autocast(device.type, enabled=False):
                loss = elbo.differentiable_loss(
                    model, guide, **_forward_kwargs(batch)
                )
        else:
            with torch.amp.autocast(
                device.type, enabled=use_amp, dtype=amp_dtype
            ):
                output = model(**_forward_kwargs(batch))
                loss = torch.nn.functional.mse_loss(
                    output["mean"], batch["cognition"]
                )
        timer.stop("forward")

        # ── Backward ──
        timer.start()
        optimizer.zero_grad()
        loss.backward()
        timer.stop("backward")

        # ── Grad clip + optimizer step ──
        timer.start()
        if clip_val:
            torch.nn.utils.clip_grad_norm_(params_for_clip, clip_val)
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        timer.stop("optimizer_step")

        # ── Total step (for cross-check) ──
        # Sum of phases for this step
        step_phases = ["data_load+transfer", "forward", "backward", "optimizer_step"]
        step_total = sum(timer.timings[p][-1] for p in step_phases)
        timer.timings["total_step"].append(step_total)

        logger.info(
            "Step %d/%d — loss: %.4f — step_time: %.0f ms",
            step_idx + 1, total_steps, loss.item(), step_total,
        )

    # ── Results ──────────────────────────────────────────────────────────────
    summary = timer.summary(skip_first=args.warmup_steps)
    print(summary)

    # GPU memory summary
    if device.type == "cuda":
        peak_gb = torch.cuda.max_memory_allocated() / 1e9
        reserved_gb = torch.cuda.max_memory_reserved() / 1e9
        print(f"\nGPU Memory:")
        print(f"  After setup:  {mem_after_setup:.2f} GB")
        print(f"  Peak alloc:   {peak_gb:.2f} GB")
        print(f"  Peak reserved: {reserved_gb:.2f} GB")

    # Save to file
    summary_path = output_dir / "timing_summary.txt"
    with open(summary_path, "w") as f:
        f.write(summary)
        if device.type == "cuda":
            f.write(f"\n\nGPU Memory:\n")
            f.write(f"  After setup:  {mem_after_setup:.2f} GB\n")
            f.write(f"  Peak alloc:   {peak_gb:.2f} GB\n")
            f.write(f"  Peak reserved: {reserved_gb:.2f} GB\n")
    print(f"\nSaved to {summary_path}")

    # Per-step breakdown
    has_edges = len(edge_counts_per_step) == total_steps
    print(f"\n{'='*90}")
    print("PER-STEP BREAKDOWN (ms)")
    print(f"{'='*90}")
    header = (
        f"{'Step':>5} {'Data':>10} {'Forward':>10} {'Backward':>10} "
        f"{'Optim':>10} {'Total':>10}"
    )
    if has_edges:
        header += f" {'MaxEdges':>10}"
    print(header)
    print("-" * 90)
    for i in range(total_steps):
        marker = " *" if i < args.warmup_steps else ""
        data_t = timer.timings["data_load+transfer"][i]
        fwd_t = timer.timings["forward"][i]
        bwd_t = timer.timings["backward"][i]
        opt_t = timer.timings["optimizer_step"][i]
        tot_t = timer.timings["total_step"][i]
        line = (
            f"{i+1:>5} {data_t:>10.0f} {fwd_t:>10.0f} {bwd_t:>10.0f} "
            f"{opt_t:>10.0f} {tot_t:>10.0f}"
        )
        if has_edges:
            line += f" {edge_counts_per_step[i]:>10,}"
        print(f"{line}{marker}")
    print("(* = warmup step, excluded from averages)")

    # Edge count correlation
    if has_edges:
        import numpy as np
        skip = args.warmup_steps
        meas_edges = np.array(edge_counts_per_step[skip:])
        meas_times = np.array(timer.timings["total_step"][skip:])
        if len(meas_edges) > 2:
            corr = np.corrcoef(meas_edges, meas_times)[0, 1]
            print(f"\nCorrelation (max_edges vs step_time): {corr:.3f}")

    print(f"\nFor kernel-level profiling, use NVIDIA Nsight Systems:")
    print(f"  nsys profile -o outputs/profiling/nsys_report \\")
    print(f"    .venv/bin/python scripts/profile_training.py \\")
    print(f"    --config configs/default.yaml \\")
    print(f"    --splits-path outputs/splits.json \\")
    print(f"    --precomputed-dir data/precomputed/rosmap/ \\")
    print(f"    --n-steps 3 --warmup-steps 1")


def _prototype_guide(model, guide, elbo, config, device):
    """Run one dummy forward pass so AutoDiagonalNormal creates its parameters."""
    from src.data.constants import N_REGIONS

    model_cfg = config.model
    dummy = {
        "region_pseudobulk": torch.zeros(
            1, N_REGIONS, model_cfg.n_cell_types, model_cfg.n_genes, device=device
        ),
        "region_mask": torch.ones(1, N_REGIONS, dtype=torch.bool, device=device),
        "cells": torch.zeros(
            1, model_cfg.n_cell_types, 1, model_cfg.n_genes, device=device
        ),
        "cell_mask": torch.ones(
            1, model_cfg.n_cell_types, 1, dtype=torch.bool, device=device
        ),
        "cell_type_mask": torch.ones(
            1, model_cfg.n_cell_types, dtype=torch.bool, device=device
        ),
        "pathology": torch.zeros(
            1,
            model_cfg.get("pathology_attention", {}).get("n_pathology_features", 3),
            device=device,
        ),
        "ccc_edge_index": torch.zeros(1, 2, 0, dtype=torch.long, device=device),
        "ccc_edge_type": torch.zeros(1, 0, dtype=torch.long, device=device),
        "ccc_edge_attr": torch.zeros(1, 0, 1, device=device),
        "ccc_edge_counts": torch.zeros(1, dtype=torch.long, device=device),
        "cognition": torch.zeros(1, 1, device=device),
    }
    with torch.no_grad():
        elbo.differentiable_loss(model, guide, **_forward_kwargs(dummy))
    n_params = sum(1 for _ in guide.parameters())
    logger.info("Guide prototyped: %d parameter tensors", n_params)


if __name__ == "__main__":
    main()
