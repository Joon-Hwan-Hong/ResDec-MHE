"""Profile DDP training to measure multi-GPU scaling and communication overhead.

Runs a short training loop on 1 or 2 GPUs using PyTorch DDP, measuring:
- Per-rank compute vs communication time (via DDP comm hooks)
- Gradient all-reduce overhead per step
- Scaling efficiency: throughput_N_gpu / (N * throughput_1_gpu)
- DDP memory tax (gradient buffers, communication overhead)
- Per-rank load imbalance (step time variance across ranks)

Launch with torchrun:
    # 1-GPU baseline:
    CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 \
        scripts/profile_ddp.py \
        --splits-path outputs/splits.json \
        --precomputed-dir data/precomputed/rosmap/

    # 2-GPU DDP:
    torchrun --nproc_per_node=2 \
        scripts/profile_ddp.py \
        --splits-path outputs/splits.json \
        --precomputed-dir data/precomputed/rosmap/

    # Compare: run 1-GPU then 2-GPU, script saves to separate output files.
"""

import argparse
import logging
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

import pyro
import pyro.poutine
from pyro.infer import TraceMeanField_ELBO
from pyro.infer.autoguide import AutoDiagonalNormal

from src.data.collate import collate_for_hgt_multiregion
from src.data.datamodule import CognitiveResilienceDataModule
from src.data.prefetch import ThreadedPrefetcher
from src.data.splits import load_splits
from src.models.full_model import build_model_from_config
from src.utils.config import load_config, validate_config
from src.utils.reproducibility import set_seed

logger = logging.getLogger(__name__)


# ── Timing utilities ──────────────────────────────────────────────────────────


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

    def stop(self, label: str) -> float:
        if self.use_cuda:
            end_event = torch.cuda.Event(enable_timing=True)
            end_event.record()
            torch.cuda.synchronize()
            elapsed_ms = self._start_event.elapsed_time(end_event)
        else:
            elapsed_ms = (time.perf_counter() - self._start_wall) * 1000.0
        self.timings[label].append(elapsed_ms)
        return elapsed_ms


class CommTimer:
    """Measures DDP all-reduce communication time via comm hooks.

    Registers a hook on each DDP gradient bucket that records elapsed time
    for the NCCL all-reduce. This captures the actual communication cost
    without requiring manual barrier insertion.
    """

    def __init__(self):
        self._comm_times_ms: list[float] = []
        self._step_comm_ms: float = 0.0
        self.per_step_comm_ms: list[float] = []
        self._n_buckets_per_step: list[int] = []
        self._bucket_count: int = 0

    def hook(self, state, bucket):
        """DDP communication hook that times the all-reduce."""
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        fut = dist.all_reduce(bucket.buffer(), async_op=True).get_future()
        end.record()

        def callback(fut):
            torch.cuda.synchronize()
            elapsed = start.elapsed_time(end)
            self._comm_times_ms.append(elapsed)
            self._step_comm_ms += elapsed
            self._bucket_count += 1
            return fut.value()

        return fut.then(callback)

    def step_done(self):
        """Call after each optimizer.step() to record per-step totals."""
        self.per_step_comm_ms.append(self._step_comm_ms)
        self._n_buckets_per_step.append(self._bucket_count)
        self._step_comm_ms = 0.0
        self._bucket_count = 0

    def summary(self, skip_first: int = 0) -> dict:
        comm = self.per_step_comm_ms[skip_first:]
        buckets = self._n_buckets_per_step[skip_first:]
        if not comm:
            return {}
        arr = np.array(comm)
        return {
            "comm_mean_ms": arr.mean(),
            "comm_std_ms": arr.std(),
            "comm_min_ms": arr.min(),
            "comm_max_ms": arr.max(),
            "buckets_per_step": int(np.mean(buckets)) if buckets else 0,
            "n_steps": len(comm),
        }


# ── Forward kwargs helper ─────────────────────────────────────────────────────


def _forward_kwargs(batch: dict) -> dict:
    kwargs = dict(
        region_pseudobulk=batch.get("region_pseudobulk"),
        region_mask=batch.get("region_mask"),
        pseudobulk=batch.get("pseudobulk"),
        ccc_edge_index=batch.get("ccc_edge_index"),
        ccc_edge_type=batch.get("ccc_edge_type"),
        ccc_edge_attr=batch.get("ccc_edge_attr"),
        cell_type_mask=batch.get("cell_type_mask"),
        pathology=batch.get("pathology"),
        cognition=batch.get("cognition"),
    )
    if "cell_data" in batch:
        kwargs["cell_data"] = batch["cell_data"]
        kwargs["cell_offsets"] = batch["cell_offsets"]
    else:
        kwargs["cells"] = batch.get("cells")
        kwargs["cell_mask"] = batch.get("cell_mask")
    return kwargs


def _move_batch_to_device(batch: dict, device: torch.device) -> dict:
    moved = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            moved[k] = v.to(device, non_blocking=True)
        else:
            moved[k] = v
    return moved


# ── Prototype guide ───────────────────────────────────────────────────────────


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
        "ccc_edge_index": torch.zeros(2, 0, dtype=torch.long, device=device),
        "ccc_edge_type": torch.zeros(0, dtype=torch.long, device=device),
        "ccc_edge_attr": torch.zeros(0, 1, device=device),
        "cognition": torch.zeros(1, 1, device=device),
    }
    with torch.no_grad():
        elbo.differentiable_loss(model, guide, **_forward_kwargs(dummy))
    n_params = sum(1 for _ in guide.parameters())
    logger.info("Guide prototyped: %d parameter tensors", n_params)


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Profile DDP training with communication timing"
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
        "--n-steps", type=int, default=10,
        help="Total training steps to run",
    )
    parser.add_argument(
        "--warmup-steps", type=int, default=3,
        help="Steps to exclude from timing averages",
    )
    parser.add_argument(
        "--num-workers", type=int, default=0,
        help="DataLoader workers per rank (0 for preloaded precomputed data)",
    )
    parser.add_argument(
        "--profile-subset", type=str, default=None,
        help="Path to profile_subset.json for reproducible subject selection",
    )
    parser.add_argument(
        "--prefetch", action="store_true",
        help="Use threaded prefetcher to overlap collation with GPU compute",
    )
    parser.add_argument(
        "overrides", nargs="*",
        help="Config overrides in dotlist format",
    )
    args = parser.parse_args()

    # ── DDP init ─────────────────────────────────────────────────────────────
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    # Only rank 0 logs
    logging.basicConfig(
        level=logging.INFO if rank == 0 else logging.WARNING,
        format=f"%(asctime)s [rank {rank}] %(name)s %(levelname)s: %(message)s",
    )

    output_dir = Path(args.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "DDP profiling: world_size=%d, rank=%d, device=%s",
        world_size, rank, device,
    )

    # ── Config ───────────────────────────────────────────────────────────────
    config = load_config(args.config, overrides=args.overrides)
    validate_config(
        config, required_keys=["experiment", "data", "model", "training", "paths"]
    )

    seed = config.experiment.get("seed", 42)
    set_seed(seed, deterministic=False, benchmark=True)

    # ── Model ────────────────────────────────────────────────────────────────
    is_bayesian = config.model.head.type == "bayesian"

    # Must set before model creation: AutoDiagonalNormal auto-converts the
    # model to a PyroModule, which needs param_state on its _Context.
    if is_bayesian:
        pyro.settings.set(module_local_params=True)

    model = build_model_from_config(config.model).to(device)

    # Wrap in DDP — tensorized HGT has no unused params
    model_ddp = DDP(model, device_ids=[local_rank])

    guide = None
    elbo = None
    if is_bayesian:
        pyro.clear_param_store()
        guide = AutoDiagonalNormal(model)
        pyro.enable_validation(False)
        elbo = TraceMeanField_ELBO()
        _prototype_guide(model, guide, elbo, config, device)

    # ── Communication hooks ──────────────────────────────────────────────────
    comm_timer = CommTimer()
    if world_size > 1:
        model_ddp.register_comm_hook(state=None, hook=comm_timer.hook)

    # ── Optimizer ────────────────────────────────────────────────────────────
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

    # ── Data ─────────────────────────────────────────────────────────────────
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

    sampler = DistributedSampler(
        train_ds, num_replicas=world_size, rank=rank, shuffle=True, seed=seed,
    )

    batch_size = config.data.dataloader.batch_size
    num_workers = args.num_workers
    prefetch = config.data.dataloader.get("prefetch_factor", 2) if num_workers > 0 else None

    train_dl = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True and num_workers > 0,
        collate_fn=collate_for_hgt_multiregion,
        persistent_workers=num_workers > 0,
        prefetch_factor=prefetch,
    )

    # ── GPU memory baseline ──────────────────────────────────────────────────
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    mem_after_setup = torch.cuda.memory_allocated(device) / 1e9

    # ── Training loop with timing ────────────────────────────────────────────
    timer = CUDATimer(device)
    model_ddp.train()
    total_steps = args.n_steps
    logger.info(
        "Running %d steps (%d warmup + %d measured) on %d GPU(s)",
        total_steps, args.warmup_steps, total_steps - args.warmup_steps, world_size,
    )

    if args.prefetch:
        logger.info("Threaded prefetcher enabled (prefetch_count=2)")
        data_source = ThreadedPrefetcher(train_dl, device, prefetch_count=2)
    else:
        data_source = train_dl

    batch_iter = iter(data_source)
    edge_counts_per_step: list[int] = []

    for step_idx in range(total_steps):
        # ── Data loading (or wait for prefetched batch) ──
        timer.start()
        try:
            batch = next(batch_iter)
        except StopIteration:
            sampler.set_epoch(step_idx)
            batch_iter = iter(data_source)
            batch = next(batch_iter)
        if not args.prefetch:
            batch = _move_batch_to_device(batch, device)
        torch.cuda.synchronize(device)
        timer.stop("data_load+transfer")

        # Record max edge count for this batch (explains variance)
        if "ccc_edge_index" in batch:
            edge_counts_per_step.append(batch["ccc_edge_index"].shape[1])

        # ── Forward ──
        timer.start()
        if is_bayesian:
            with torch.amp.autocast(device.type, enabled=False):
                loss = elbo.differentiable_loss(
                    model, guide, **_forward_kwargs(batch)
                )
        else:
            with torch.amp.autocast(device.type, enabled=True, dtype=torch.bfloat16):
                output = model_ddp(**_forward_kwargs(batch))
                loss = torch.nn.functional.mse_loss(
                    output["mean"], batch["cognition"]
                )
        timer.stop("forward")

        # ── Backward (includes DDP gradient sync) ──
        timer.start()
        optimizer.zero_grad()
        loss.backward()
        timer.stop("backward+comm")

        # Record comm timing for this step
        if world_size > 1:
            comm_timer.step_done()

        # ── Grad clip + optimizer step ──
        timer.start()
        if clip_val:
            torch.nn.utils.clip_grad_norm_(params_for_clip, clip_val)
        optimizer.step()
        torch.cuda.synchronize(device)
        timer.stop("optimizer_step")

        # Step total
        step_phases = ["data_load+transfer", "forward", "backward+comm", "optimizer_step"]
        step_total = sum(timer.timings[p][-1] for p in step_phases)
        timer.timings["total_step"].append(step_total)

        if rank == 0:
            max_edges = edge_counts_per_step[-1] if edge_counts_per_step else 0
            logger.info(
                "Step %d/%d — loss: %.4f — step_time: %.0f ms — max_edges: %d",
                step_idx + 1, total_steps, loss.item(), step_total, max_edges,
            )

    # ── Gather per-rank timings ──────────────────────────────────────────────
    # Each rank sends its measured step times to rank 0 for imbalance analysis
    skip = args.warmup_steps
    measured_steps = timer.timings["total_step"][skip:]
    local_times = torch.tensor(measured_steps, device=device, dtype=torch.float64)

    if world_size > 1:
        # Pad to same length across ranks
        max_len = torch.tensor(len(local_times), device=device)
        dist.all_reduce(max_len, op=dist.ReduceOp.MAX)
        padded = torch.zeros(int(max_len.item()), device=device, dtype=torch.float64)
        padded[:len(local_times)] = local_times

        all_times = [torch.zeros_like(padded) for _ in range(world_size)]
        dist.all_gather(all_times, padded)
    else:
        all_times = [local_times]

    # ── Report (rank 0 only) ─────────────────────────────────────────────────
    if rank == 0:
        peak_gb = torch.cuda.max_memory_allocated(device) / 1e9
        reserved_gb = torch.cuda.max_memory_reserved(device) / 1e9

        lines = []
        lines.append(f"\n{'='*90}")
        lines.append(f"DDP PROFILING REPORT — {world_size} GPU(s)")
        lines.append(f"{'='*90}")

        # ── Phase timing ──
        lines.append(f"\n{'─'*90}")
        lines.append("PHASE TIMING (rank 0, milliseconds)")
        lines.append(f"  (first {skip} steps excluded as warmup)")
        lines.append(f"{'─'*90}")
        lines.append(
            f"{'Phase':<25} {'Mean':>10} {'Std':>10} {'Min':>10} "
            f"{'Max':>10} {'Count':>7}"
        )
        lines.append("-" * 90)

        total_compute = 0.0
        for label in ["data_load+transfer", "forward", "backward+comm", "optimizer_step", "total_step"]:
            t = timer.timings[label][skip:]
            if not t:
                continue
            arr = np.array(t)
            if label not in ("total_step",):
                total_compute += arr.mean()
            lines.append(
                f"{label:<25} {arr.mean():>10.1f} {arr.std():>10.1f} "
                f"{arr.min():>10.1f} {arr.max():>10.1f} {len(t):>7}"
            )
        lines.append("-" * 90)
        lines.append(f"{'sum of phases':<25} {total_compute:>10.1f}")

        # ── Communication overhead ──
        if world_size > 1:
            comm_stats = comm_timer.summary(skip_first=skip)
            lines.append(f"\n{'─'*90}")
            lines.append("COMMUNICATION OVERHEAD (NCCL all-reduce)")
            lines.append(f"{'─'*90}")
            if comm_stats:
                bwd = timer.timings["backward+comm"][skip:]
                bwd_mean = np.mean(bwd)
                comm_mean = comm_stats["comm_mean_ms"]
                compute_in_bwd = bwd_mean - comm_mean
                comm_pct = (comm_mean / bwd_mean * 100) if bwd_mean > 0 else 0

                lines.append(f"  All-reduce mean:     {comm_mean:>10.1f} ms")
                lines.append(f"  All-reduce std:      {comm_stats['comm_std_ms']:>10.1f} ms")
                lines.append(f"  All-reduce min:      {comm_stats['comm_min_ms']:>10.1f} ms")
                lines.append(f"  All-reduce max:      {comm_stats['comm_max_ms']:>10.1f} ms")
                lines.append(f"  Buckets per step:    {comm_stats['buckets_per_step']:>10d}")
                lines.append(f"  backward+comm total: {bwd_mean:>10.1f} ms")
                lines.append(f"  Compute in backward: {compute_in_bwd:>10.1f} ms")
                lines.append(f"  Comm fraction:       {comm_pct:>9.1f}%")

                # Overlap analysis: if comm < backward_compute, comm is hidden
                if comm_mean < compute_in_bwd:
                    lines.append(f"  Overlap status:      FULLY OVERLAPPED (comm hidden behind compute)")
                else:
                    exposed = comm_mean - compute_in_bwd
                    lines.append(f"  Overlap status:      PARTIALLY EXPOSED ({exposed:.1f} ms exposed)")
            else:
                lines.append("  No communication data (single step?)")

        # ── Scaling efficiency ──
        lines.append(f"\n{'─'*90}")
        lines.append("SCALING & THROUGHPUT")
        lines.append(f"{'─'*90}")
        batch_size = config.data.dataloader.batch_size
        step_mean = np.mean(measured_steps) if measured_steps else 1.0
        samples_per_sec = (batch_size * world_size) / (step_mean / 1000.0)
        lines.append(f"  World size:          {world_size}")
        lines.append(f"  Batch size per GPU:  {batch_size}")
        lines.append(f"  Effective batch:     {batch_size * world_size}")
        lines.append(f"  Mean step time:      {step_mean:.1f} ms")
        lines.append(f"  Throughput:          {samples_per_sec:.1f} samples/sec")
        lines.append(f"  Per-GPU throughput:  {samples_per_sec / world_size:.1f} samples/sec/GPU")

        # ── Load imbalance ──
        if world_size > 1:
            lines.append(f"\n{'─'*90}")
            lines.append("LOAD IMBALANCE (per-rank step times)")
            lines.append(f"{'─'*90}")
            n_measured = len(measured_steps)
            for r in range(world_size):
                rank_times = all_times[r][:n_measured].cpu().numpy()
                lines.append(
                    f"  Rank {r}: mean={rank_times.mean():.1f} ms, "
                    f"std={rank_times.std():.1f} ms, "
                    f"min={rank_times.min():.1f} ms, max={rank_times.max():.1f} ms"
                )

            # Imbalance metric: max deviation from mean across ranks
            rank_means = [all_times[r][:n_measured].cpu().mean().item() for r in range(world_size)]
            overall_mean = np.mean(rank_means)
            max_deviation = max(abs(m - overall_mean) for m in rank_means)
            imbalance_pct = (max_deviation / overall_mean * 100) if overall_mean > 0 else 0
            lines.append(f"  Imbalance:           {imbalance_pct:.1f}% (max rank deviation from mean)")

        # ── Edge count correlation ──
        if edge_counts_per_step:
            lines.append(f"\n{'─'*90}")
            lines.append("BATCH EDGE COUNT vs STEP TIME (explains variance)")
            lines.append(f"{'─'*90}")
            lines.append(f"  {'Step':>5} {'MaxEdges':>10} {'StepTime':>10}")
            for i in range(len(edge_counts_per_step)):
                marker = " *" if i < skip else ""
                step_t = timer.timings["total_step"][i]
                lines.append(
                    f"  {i+1:>5} {edge_counts_per_step[i]:>10,} {step_t:>10.0f}{marker}"
                )
            # Correlation
            meas_edges = np.array(edge_counts_per_step[skip:])
            meas_times = np.array(timer.timings["total_step"][skip:])
            if len(meas_edges) > 2:
                corr = np.corrcoef(meas_edges, meas_times)[0, 1]
                lines.append(f"  Correlation (edges vs time): {corr:.3f}")

        # ── Memory ──
        lines.append(f"\n{'─'*90}")
        lines.append("GPU MEMORY (rank 0)")
        lines.append(f"{'─'*90}")
        lines.append(f"  After setup:   {mem_after_setup:.2f} GB")
        lines.append(f"  Peak alloc:    {peak_gb:.2f} GB")
        lines.append(f"  Peak reserved: {reserved_gb:.2f} GB")
        if world_size > 1:
            # DDP adds gradient buffers ≈ model_size
            model_size_gb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e9
            lines.append(f"  Model params:  {model_size_gb:.3f} GB")
            lines.append(f"  DDP grad buf:  ~{model_size_gb:.3f} GB (estimated)")

        lines.append(f"\n{'='*90}")

        report = "\n".join(lines)
        print(report)

        # Save report
        suffix = f"{world_size}gpu"
        summary_path = output_dir / f"ddp_profile_{suffix}.txt"
        with open(summary_path, "w") as f:
            f.write(report)
        print(f"\nSaved to {summary_path}")

    # ── Cleanup ──────────────────────────────────────────────────────────────
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
