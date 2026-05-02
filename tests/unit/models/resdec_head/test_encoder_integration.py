"""Integration smoke test: current CognitiveResilienceModel produces a
subject embedding consumable by the new head stack under P5.

Loads from configs/default.yaml, runs a small dummy batch on CPU (or the
default device), asserts the 'attended' key has the expected shape.
"""
import pytest
import torch
from omegaconf import OmegaConf

from src.models.full_model import build_model_from_config
from tests.conftest import WORKTREE_ROOT, _build_canonical_batch


@pytest.fixture(scope="module")
def encoder():
    cfg = OmegaConf.load(WORKTREE_ROOT / "configs" / "default.yaml")
    OmegaConf.set_struct(cfg.model, False)
    cfg.model.n_genes = 4785
    cfg.model.n_cell_types = 31
    return build_model_from_config(cfg.model)


def _make_dummy_batch(batch_size: int, device: torch.device) -> dict:
    """Build a dummy batch: mix of PFC-only and multi-region subjects.

    Routes through the shared canonical batch builder. Note: this batch
    differs from a uniform-PFC batch by re-flagging multi-region subjects
    after construction; this matches the original test's behaviour of
    "mix of PFC-only and multi-region subjects."
    """
    batch = _build_canonical_batch(
        batch_size=batch_size,
        n_genes=4785,
        cells_per_ct=10,
        edges_per_subj=50,
        device=device,
        seed=0,
        pfc_only=True,
    )
    # Flip first ⌈B/8⌉ subjects to multi-region (preserves original behaviour)
    region_mask = batch["region_mask"]
    for i in range(max(1, batch_size // 8)):
        region_mask[i, :] = True
    batch["region_mask"] = region_mask
    return batch

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
