"""Integration smoke test: current CognitiveResilienceModel produces a
subject embedding consumable by the new head stack under P5.

Loads from configs/default.yaml, runs a small dummy batch on CPU (or the
default device), asserts the 'attended' key has the expected shape.
"""
import pytest
import torch
from omegaconf import OmegaConf

from src.models.full_model import build_model_from_config


@pytest.fixture(scope="module")
def encoder():
    cfg = OmegaConf.load("configs/default.yaml")
    OmegaConf.set_struct(cfg.model, False)
    cfg.model.n_genes = 4785
    cfg.model.n_cell_types = 31
    return build_model_from_config(cfg.model)


def _make_dummy_batch(batch_size: int, device: torch.device) -> dict:
    """Mirror bench_p5_full_model.make_dummy_batch: mix of PFC-only and multi-region."""
    N_CT = 31
    N_GENES = 4785
    N_REGIONS = 6
    rng = torch.Generator(device="cpu").manual_seed(0)

    region_mask = torch.zeros(batch_size, N_REGIONS, dtype=torch.bool)
    region_mask[:, 0] = True
    for i in range(max(1, batch_size // 8)):
        region_mask[i, :] = True

    region_pseudobulk = torch.randn(batch_size, N_REGIONS, N_CT, N_GENES, generator=rng)
    region_pseudobulk = region_pseudobulk * region_mask.float().unsqueeze(-1).unsqueeze(-1)

    edges_per_subj = 50
    total_edges = batch_size * edges_per_subj
    ccc_edge_index = torch.randint(0, N_CT, (2, total_edges), generator=rng)
    ccc_edge_type = torch.randint(0, 5, (total_edges,), generator=rng)
    ccc_edge_attr = torch.rand(total_edges, 1, generator=rng)

    cells_per_ct = 10
    cells_per_subject = cells_per_ct * N_CT
    total_cells = batch_size * cells_per_subject
    cell_data = torch.randn(total_cells, N_GENES, generator=rng)
    offsets_per_subj = torch.arange(0, cells_per_subject + 1, cells_per_ct, dtype=torch.long)
    subj_offsets = torch.arange(batch_size, dtype=torch.long) * cells_per_subject
    cell_offsets = subj_offsets.unsqueeze(1) + offsets_per_subj.unsqueeze(0)

    return {
        "region_pseudobulk": region_pseudobulk.to(device),
        "region_mask": region_mask.to(device),
        "ccc_edge_index": ccc_edge_index.to(device),
        "ccc_edge_type": ccc_edge_type.to(device),
        "ccc_edge_attr": ccc_edge_attr.to(device),
        "cell_type_mask": torch.ones(batch_size, N_CT, dtype=torch.bool).to(device),
        "cell_data": cell_data.to(device),
        "cell_offsets": cell_offsets.to(device),
        "pathology": torch.randn(batch_size, 3, generator=rng).to(device),
        "cognition": torch.randn(batch_size, 1, generator=rng).to(device),
    }


def test_encoder_loads_from_config(encoder):
    assert encoder is not None
    n_params = sum(p.numel() for p in encoder.parameters())
    assert n_params > 1_000_000  # Current model ~2.5M params


def test_encoder_forward_produces_attended_embedding(encoder):
    """Encoder.forward returns dict with 'attended' [B, d_subject] we can route to head."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder.to(device).eval()
    batch = _make_dummy_batch(batch_size=2, device=device)
    with torch.no_grad():
        out = encoder(
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
    assert "attended" in out
    assert out["attended"].ndim == 2
    assert out["attended"].shape[0] == 2  # batch
    # d_subject is d_embed * 2 per default config (32 * 2 = 64)
    assert out["attended"].shape[1] >= 32  # generous lower bound


def test_encoder_backward_runs(encoder):
    """Gradient flows through the encoder when using 'mean' output for supervision."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder.to(device).train()
    batch = _make_dummy_batch(batch_size=2, device=device)
    out = encoder(
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
    # At least some encoder params should have gradients
    has_grad = any(p.grad is not None and p.grad.abs().max() > 0
                   for p in encoder.parameters() if p.requires_grad)
    assert has_grad
