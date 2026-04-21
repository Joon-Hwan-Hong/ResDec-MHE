"""Benchmark the actual current full_model.py (P5 baseline reference).

Loads the real CognitiveResilienceModel via build_model_from_config, constructs
a representative batch matching what the existing collate produces, and times
forward+backward on GPU 0.

This gives a true P5 wall-clock number — not a simplified mock.
"""
from __future__ import annotations
import time
import torch
from omegaconf import OmegaConf

from src.models.full_model import build_model_from_config

BATCH = 24
N_WARMUP = 3
N_ITER = 10
N_CT = 31
N_GENES = 4785
N_REGIONS = 6


def make_dummy_batch(batch_size: int, device: torch.device) -> dict:
    """Construct a dummy batch matching the existing collate output shapes.

    Mixes single-region (most) with a few multi-region subjects, matching the
    real 87.6% PFC-only / 8.3% all-6-region distribution.
    """
    rng = torch.Generator(device="cpu").manual_seed(0)

    # 87.5% single-region, 12.5% multi-region in this small batch
    region_mask = torch.zeros(batch_size, N_REGIONS, dtype=torch.bool)
    region_mask[:, 0] = True
    for i in range(batch_size // 8):
        region_mask[i, :] = True

    region_pseudobulk = torch.randn(
        batch_size, N_REGIONS, N_CT, N_GENES, generator=rng
    )
    region_pseudobulk = region_pseudobulk * region_mask.float().unsqueeze(-1).unsqueeze(-1)

    # CCC edges: use a modest fixed edge count per subject
    edges_per_subj = 50
    total_edges = batch_size * edges_per_subj
    ccc_edge_index = torch.randint(0, N_CT, (2, total_edges), generator=rng)
    ccc_edge_type = torch.randint(0, 5, (total_edges,), generator=rng)
    ccc_edge_attr = torch.rand(total_edges, 1, generator=rng)

    cell_type_mask = torch.ones(batch_size, N_CT, dtype=torch.bool)
    pathology = torch.randn(batch_size, 3, generator=rng)

    # Cell-level: ~50 cells per (subject, cell_type), flat concatenated
    cells_per_ct = 10  # keep modest to limit memory in benchmark
    cells_per_subject = cells_per_ct * N_CT
    total_cells = batch_size * cells_per_subject
    cell_data = torch.randn(total_cells, N_GENES, generator=rng)
    # cell_offsets: [B, N_CT + 1] cumulative offsets per subject
    offsets_per_subj = torch.arange(0, cells_per_subject + 1, cells_per_ct, dtype=torch.long)
    subj_offsets = torch.arange(batch_size, dtype=torch.long) * cells_per_subject
    cell_offsets = subj_offsets.unsqueeze(1) + offsets_per_subj.unsqueeze(0)  # [B, N_CT+1]

    # Move to device
    return {
        "region_pseudobulk": region_pseudobulk.to(device),
        "region_mask": region_mask.to(device),
        "ccc_edge_index": ccc_edge_index.to(device),
        "ccc_edge_type": ccc_edge_type.to(device),
        "ccc_edge_attr": ccc_edge_attr.to(device),
        "cell_type_mask": cell_type_mask.to(device),
        "cell_data": cell_data.to(device),
        "cell_offsets": cell_offsets.to(device),
        "pathology": pathology.to(device),
        "cognition": torch.randn(batch_size, 1, generator=rng).to(device),
    }


def main():
    device = torch.device("cuda:0")

    cfg = OmegaConf.load("configs/default.yaml")
    # Set required n_genes / n_cell_types that default config leaves unset
    OmegaConf.set_struct(cfg.model, False)
    cfg.model.n_genes = N_GENES
    cfg.model.n_cell_types = N_CT
    # Ensure we build with sensible defaults
    print(f"Building model from configs/default.yaml (n_genes={N_GENES}, n_cell_types={N_CT})")
    model = build_model_from_config(cfg.model)
    model = model.to(device)
    model.train()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,} ({n_params/1e6:.2f}M)")

    batch = make_dummy_batch(BATCH, device)

    # Warmup
    for _ in range(N_WARMUP):
        out = model(
            region_pseudobulk=batch["region_pseudobulk"],
            region_mask=batch["region_mask"],
            ccc_edge_index=batch["ccc_edge_index"],
            ccc_edge_type=batch["ccc_edge_type"],
            ccc_edge_attr=batch["ccc_edge_attr"],
            cell_type_mask=batch["cell_type_mask"],
            cell_data=batch["cell_data"],
            cell_offsets=batch["cell_offsets"],
            pathology=batch["pathology"],
            cognition=batch["cognition"],
        )
        loss = out["mean"].pow(2).mean()
        loss.backward()
        model.zero_grad(set_to_none=True)

    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)

    fwd_times_ms = []
    bwd_times_ms = []
    for _ in range(N_ITER):
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        out = model(
            region_pseudobulk=batch["region_pseudobulk"],
            region_mask=batch["region_mask"],
            ccc_edge_index=batch["ccc_edge_index"],
            ccc_edge_type=batch["ccc_edge_type"],
            ccc_edge_attr=batch["ccc_edge_attr"],
            cell_type_mask=batch["cell_type_mask"],
            cell_data=batch["cell_data"],
            cell_offsets=batch["cell_offsets"],
            pathology=batch["pathology"],
            cognition=batch["cognition"],
        )
        torch.cuda.synchronize(device)
        t1 = time.perf_counter()
        loss = out["mean"].pow(2).mean()
        loss.backward()
        torch.cuda.synchronize(device)
        t2 = time.perf_counter()
        fwd_times_ms.append((t1 - t0) * 1000)
        bwd_times_ms.append((t2 - t1) * 1000)
        model.zero_grad(set_to_none=True)

    peak_gb = torch.cuda.max_memory_allocated(device) / 1e9

    fwd = torch.tensor(fwd_times_ms)
    bwd = torch.tensor(bwd_times_ms)
    total = fwd + bwd

    print(f"\nCurrent full_model.py (P5 reference, batch={BATCH}):")
    print(f"  params: {n_params/1e6:.2f}M")
    print(f"  forward: {fwd.mean():.2f} ± {fwd.std():.2f} ms")
    print(f"  backward: {bwd.mean():.2f} ± {bwd.std():.2f} ms")
    print(f"  step (fwd+bwd): {total.mean():.2f} ± {total.std():.2f} ms")
    print(f"  peak GPU memory: {peak_gb:.2f} GB")

    steps_per_fold = 60 * 17
    total_steps = steps_per_fold * 5 * 11 * 2
    print(f"\n  per fold first-layer time: {total.mean().item() * steps_per_fold / 1000 / 60:.2f} min")
    print(f"  full eval (5 folds × 11 ablations × 2 seeds, {total_steps:,} steps): "
          f"{total.mean().item() * total_steps / 1000 / 3600:.2f} hours")


if __name__ == "__main__":
    main()
