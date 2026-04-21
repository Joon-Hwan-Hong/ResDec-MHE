"""Microbenchmark: compare first-layer compute cost of encoder options.

Options:
  A       — flat 895,173-feature input → BatchEnsemble Linear(895K, 128, k=8)
  B-lean  — flat 153,198-feature input → BatchEnsemble Linear(153K, 128, k=8)
  P5-like — structured [31, 4785] → shared gene gate + Linear(4785, d_ct=64)
            + flatten → BatchEnsemble Linear(31*64, 128, k=8)

Measures forward + backward wall-clock and peak GPU memory for a single batch
on CUDA 0. Warms up, then runs N iterations, reports mean/std.

Intent: give honest compute numbers so the architecture choice is informed by
data, not rhetoric.
"""
from __future__ import annotations
import time
import torch
import torch.nn as nn

BATCH = 24
N_WARMUP = 5
N_ITER = 20
D_HIDDEN = 128
K_MEMBERS = 8
N_CT = 31
N_GENES = 4785
D_CT = 64


class BatchEnsembleLinear(nn.Module):
    """TabM-style BatchEnsemble: shared W + per-member rank-1 (s, r) scaling.

    y_k = (x * s_k) @ W * r_k   for k in 0..K-1
    """

    def __init__(self, d_in: int, d_out: int, k: int):
        super().__init__()
        self.W = nn.Parameter(torch.randn(d_in, d_out) * 0.01)
        self.s = nn.Parameter(torch.randn(k, d_in) * 0.01 + 1.0)
        self.r = nn.Parameter(torch.randn(k, d_out) * 0.01 + 1.0)
        self.k = k

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: [B, d_in]
        x_expanded = x.unsqueeze(1) * self.s.unsqueeze(0)  # [B, k, d_in]
        y = x_expanded @ self.W  # [B, k, d_out]
        y = y * self.r.unsqueeze(0)  # [B, k, d_out]
        return y


class FlatEncoder(nn.Module):
    """Represents Option A or B-lean: flat input → BatchEnsemble."""

    def __init__(self, n_features: int):
        super().__init__()
        self.be = BatchEnsembleLinear(n_features, D_HIDDEN, K_MEMBERS)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.be(x)


class StructuredEncoder(nn.Module):
    """Represents P5-like: structured [B, 31, 4785] → shared gene gate +
    shared Linear(4785, d_ct) → flatten → BatchEnsemble Linear(31*d_ct, 128).
    """

    def __init__(self):
        super().__init__()
        self.gene_gate = nn.Parameter(torch.randn(N_GENES))
        self.proj = nn.Linear(N_GENES, D_CT)
        self.be = BatchEnsembleLinear(N_CT * D_CT, D_HIDDEN, K_MEMBERS)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: [B, 31, 4785]
        gate = torch.sigmoid(self.gene_gate)  # [4785]
        x = x * gate.view(1, 1, -1)  # [B, 31, 4785]
        x = self.proj(x)  # [B, 31, d_ct]
        x = x.flatten(1)  # [B, 31*d_ct]
        return self.be(x)


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def benchmark(name: str, module: nn.Module, x: torch.Tensor) -> dict:
    """Run forward+backward N_ITER times after N_WARMUP, collect stats."""
    device = torch.device("cuda:0")
    module = module.to(device)
    x = x.to(device)

    # Warmup
    for _ in range(N_WARMUP):
        out = module(x)
        loss = out.pow(2).mean()
        loss.backward()
        module.zero_grad(set_to_none=True)

    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)

    fwd_times_ms = []
    bwd_times_ms = []
    for _ in range(N_ITER):
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        out = module(x)
        torch.cuda.synchronize(device)
        t1 = time.perf_counter()
        loss = out.pow(2).mean()
        loss.backward()
        torch.cuda.synchronize(device)
        t2 = time.perf_counter()
        fwd_times_ms.append((t1 - t0) * 1000)
        bwd_times_ms.append((t2 - t1) * 1000)
        module.zero_grad(set_to_none=True)

    peak_gb = torch.cuda.max_memory_allocated(device) / 1e9

    fwd_ms = torch.tensor(fwd_times_ms)
    bwd_ms = torch.tensor(bwd_times_ms)
    total_ms = fwd_ms + bwd_ms

    return {
        "name": name,
        "n_params": count_params(module),
        "input_shape": list(x.shape),
        "fwd_ms_mean": fwd_ms.mean().item(),
        "fwd_ms_std": fwd_ms.std().item(),
        "bwd_ms_mean": bwd_ms.mean().item(),
        "bwd_ms_std": bwd_ms.std().item(),
        "step_ms_mean": total_ms.mean().item(),
        "step_ms_std": total_ms.std().item(),
        "peak_gb": peak_gb,
    }


def main():
    # Free GPU memory before each measurement
    torch.cuda.empty_cache()

    # Option A: flat 895,173 features
    torch.cuda.empty_cache()
    res_a = benchmark(
        "A (flat 895K zero-pad)",
        FlatEncoder(895173),
        torch.randn(BATCH, 895173),
    )

    # Option B-lean: flat 153,198 features
    torch.cuda.empty_cache()
    res_b = benchmark(
        "B-lean (flat 153K)",
        FlatEncoder(153198),
        torch.randn(BATCH, 153198),
    )

    # Option P5-like: structured [B, 31, 4785]
    torch.cuda.empty_cache()
    res_p5 = benchmark(
        "P5-like (structured 31x4785)",
        StructuredEncoder(),
        torch.randn(BATCH, N_CT, N_GENES),
    )

    # Print results
    print("\n" + "=" * 100)
    print(f"Microbenchmark: single forward+backward pass, batch={BATCH}, "
          f"N_WARMUP={N_WARMUP}, N_ITER={N_ITER}")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print("=" * 100)
    hdr = (
        f"{'option':35s}{'params (M)':>12s}{'fwd ms':>12s}"
        f"{'bwd ms':>12s}{'step ms':>12s}{'peak GB':>10s}"
    )
    print(hdr)
    print("-" * 100)
    for r in (res_a, res_b, res_p5):
        print(
            f"{r['name']:35s}"
            f"{r['n_params']/1e6:>11.2f}M"
            f"{r['fwd_ms_mean']:>8.2f}±{r['fwd_ms_std']:>2.1f}"
            f"{r['bwd_ms_mean']:>8.2f}±{r['bwd_ms_std']:>2.1f}"
            f"{r['step_ms_mean']:>8.2f}±{r['step_ms_std']:>2.1f}"
            f"{r['peak_gb']:>9.2f}G"
        )
    print("-" * 100)

    # Extrapolation: estimate full-training cost
    # Per prior memory: ~60 epochs × 412 train subjects / batch 24 = ~17 batches/epoch
    # 60 epochs × ~17 batches × 2 (fwd+bwd counted once) = 1020 steps per fold
    # × 5 folds × 11 ablations × 2 seeds = 112,200 steps full eval
    steps_per_fold = 60 * 17
    total_steps_full_eval = steps_per_fold * 5 * 11 * 2
    print(f"\nExtrapolation (60 epochs × 17 batches × 5 folds × 11 ablations × 2 seeds "
          f"= {total_steps_full_eval:,} steps):")
    print(f"{'option':35s}{'per fold (min)':>18s}{'full eval (hrs)':>20s}")
    print("-" * 100)
    for r in (res_a, res_b, res_p5):
        fold_min = r['step_ms_mean'] * steps_per_fold / 1000 / 60
        full_hrs = r['step_ms_mean'] * total_steps_full_eval / 1000 / 3600
        print(f"{r['name']:35s}{fold_min:>17.2f}m{full_hrs:>19.2f}h")
    print("=" * 100)
    print("\nCaveats:")
    print(" - First-layer-only benchmark: total model cost will be higher.")
    print(" - Training adds optimizer step, data loading, validation, checkpointing.")
    print(" - GPU memory for the full model may exceed these peak numbers.")


if __name__ == "__main__":
    main()
