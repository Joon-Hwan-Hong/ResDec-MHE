"""Tests for the encoder split required by counterfactual caching (P1.1).

These tests verify that the new public methods on CognitiveResilienceModel
let a caller cache the cell-branch embedding once and recompute only the
HGT-dependent portion per region_pseudobulk perturbation, while keeping
forward() bit-identical to the unmodified implementation.

Bit-identity (atol=0, rtol=0) is the load-bearing guarantee here: the d4
canonical sweep imports full_model.py via train.py, and any drift in the
canonical training path would silently shift sweep results.
"""
from __future__ import annotations

import torch
import pytest
from omegaconf import OmegaConf

from src.models.full_model import build_model_from_config


@pytest.fixture(scope="module")
def encoder():
    cfg = OmegaConf.load("configs/default.yaml")
    OmegaConf.set_struct(cfg.model, False)
    cfg.model.n_genes = 64
    cfg.model.n_cell_types = 31
    # Keep dims small for CPU speed; default architecture otherwise.
    cfg.model.d_embed = 32
    cfg.model.d_fused = 32
    if "set_transformer" in cfg.model:
        cfg.model.set_transformer.n_inducing_points = 8
        cfg.model.set_transformer.n_isab_layers = 1
    if "hgt" in cfg.model:
        cfg.model.hgt.n_layers = 1
    return build_model_from_config(cfg.model).eval()


def _make_dummy_batch(batch_size: int, n_genes: int, device: torch.device) -> dict:
    """Same shape contract as test_encoder_integration._make_dummy_batch but smaller."""
    N_CT = 31
    N_REGIONS = 6
    rng = torch.Generator(device="cpu").manual_seed(0)

    region_mask = torch.zeros(batch_size, N_REGIONS, dtype=torch.bool)
    region_mask[:, 0] = True

    region_pseudobulk = torch.randn(batch_size, N_REGIONS, N_CT, n_genes, generator=rng)
    region_pseudobulk = region_pseudobulk * region_mask.float().unsqueeze(-1).unsqueeze(-1)

    edges_per_subj = 8
    total_edges = batch_size * edges_per_subj
    ccc_edge_index = torch.randint(0, N_CT, (2, total_edges), generator=rng)
    ccc_edge_type = torch.randint(0, 5, (total_edges,), generator=rng)
    ccc_edge_attr = torch.rand(total_edges, 1, generator=rng)

    cells_per_ct = 3
    cells_per_subject = cells_per_ct * N_CT
    total_cells = batch_size * cells_per_subject
    cell_data = torch.randn(total_cells, n_genes, generator=rng)
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


def _call_forward(model, batch: dict, **extra) -> dict:
    return model(
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
        **extra,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test: compute_cell_emb_only exists and matches the cell branch in forward()
# ─────────────────────────────────────────────────────────────────────────────


def test_compute_cell_emb_only_returns_tensor_with_expected_shape(encoder):
    device = next(encoder.parameters()).device
    batch = _make_dummy_batch(batch_size=2, n_genes=encoder.n_genes, device=device)

    with torch.no_grad():
        cell_emb = encoder.compute_cell_emb_only(batch)

    assert isinstance(cell_emb, torch.Tensor)
    assert cell_emb.dim() == 3
    assert cell_emb.shape[0] == 2  # batch size
    assert cell_emb.shape[1] == encoder.n_cell_types


def test_compute_cell_emb_only_matches_internal_cell_branch(encoder):
    """compute_cell_emb_only output equals what forward() computes internally
    for the cell branch (verified via direct cell_transformer call)."""
    device = next(encoder.parameters()).device
    batch = _make_dummy_batch(batch_size=2, n_genes=encoder.n_genes, device=device)

    with torch.no_grad():
        cell_emb_via_method = encoder.compute_cell_emb_only(batch)
        cell_emb_direct, _ = encoder.cell_transformer(
            batch["cell_data"], batch["cell_offsets"], return_attention=False,
        )

    assert torch.equal(cell_emb_via_method, cell_emb_direct)


# ─────────────────────────────────────────────────────────────────────────────
# Test: forward_with_cached_cell_emb produces the same dict as forward()
# ─────────────────────────────────────────────────────────────────────────────


def test_forward_with_cached_cell_emb_matches_full_forward_bitexact(encoder):
    """forward(batch) and compute_cell_emb_only -> forward_with_cached_cell_emb
    must be BIT-IDENTICAL (atol=0, rtol=0). This is the load-bearing test.

    Uses torch.manual_seed before each forward call to anchor any pyro
    PyroSample weight sampling (BayesianPredictionHead) so the two paths
    consume the same RNG draws.
    """
    device = next(encoder.parameters()).device
    batch = _make_dummy_batch(batch_size=2, n_genes=encoder.n_genes, device=device)

    with torch.no_grad():
        torch.manual_seed(123)
        out_full = _call_forward(encoder, batch)

        torch.manual_seed(123)
        cell_emb = encoder.compute_cell_emb_only(batch)
        out_cached = encoder.forward_with_cached_cell_emb(
            batch, cell_emb,
        )

    assert set(out_cached.keys()) == set(out_full.keys()), (
        f"Output keys differ: full={sorted(out_full.keys())} "
        f"cached={sorted(out_cached.keys())}"
    )
    for k in out_full:
        v_full = out_full[k]
        v_cached = out_cached[k]
        if v_full is None:
            assert v_cached is None, f"Key {k}: full=None but cached={v_cached}"
            continue
        assert isinstance(v_cached, torch.Tensor), f"Key {k}: cached not a tensor"
        assert v_cached.shape == v_full.shape, f"Key {k}: shape mismatch"
        # Bit-exact: atol=0, rtol=0
        assert torch.equal(v_cached, v_full), (
            f"Key {k}: not bit-identical. "
            f"max abs diff = {(v_cached - v_full).abs().max().item():.3e}"
        )


def test_forward_deterministic_under_seeded_rng(encoder):
    """forward() must consume the same RNG draws on each call when seeded
    identically. This anchors the bit-identity test above: if
    BayesianPredictionHead consumes a different number of RNG draws after
    the dispatch refactor, the cached and full paths would diverge.
    Seeding before each call makes the test independent of the absolute
    pyro state and isolates the order of RNG consumption.
    """
    device = next(encoder.parameters()).device
    batch = _make_dummy_batch(batch_size=2, n_genes=encoder.n_genes, device=device)

    with torch.no_grad():
        torch.manual_seed(456)
        out1 = _call_forward(encoder, batch)
        torch.manual_seed(456)
        out2 = _call_forward(encoder, batch)

    for k in out1:
        if out1[k] is None:
            assert out2[k] is None
            continue
        assert torch.equal(out1[k], out2[k]), f"Key {k} not deterministic across two forward calls"


# ─────────────────────────────────────────────────────────────────────────────
# Test: caching enables independence from cell-branch on perturbed input
# ─────────────────────────────────────────────────────────────────────────────


def test_cached_cell_emb_invariant_to_region_pseudobulk_perturbation(encoder):
    """When region_pseudobulk changes but cell_data/cell_offsets do not,
    the cached cell_emb is correct for the new batch. This is the property
    the CF orchestrator depends on."""
    device = next(encoder.parameters()).device
    batch = _make_dummy_batch(batch_size=2, n_genes=encoder.n_genes, device=device)

    perturbed = {**batch}
    # Perturb only region_pseudobulk (the slice the CF moves)
    rng = torch.Generator(device="cpu").manual_seed(99)
    perturbed["region_pseudobulk"] = (
        batch["region_pseudobulk"] + 0.5 * torch.randn(
            batch["region_pseudobulk"].shape, generator=rng,
        ).to(device)
    )

    with torch.no_grad():
        cell_emb_orig = encoder.compute_cell_emb_only(batch)
        cell_emb_pert = encoder.compute_cell_emb_only(perturbed)

    # Cell branch is invariant to region_pseudobulk perturbation
    assert torch.equal(cell_emb_orig, cell_emb_pert), (
        "Cell branch must be independent of region_pseudobulk; got difference"
    )

    # And forward_with_cached_cell_emb on the perturbed batch using the cached
    # cell_emb should equal the full forward on the perturbed batch (with the
    # same RNG seed for the BayesianPredictionHead PyroSample draws).
    with torch.no_grad():
        torch.manual_seed(789)
        out_full_pert = _call_forward(encoder, perturbed)
        torch.manual_seed(789)
        out_cached_pert = encoder.forward_with_cached_cell_emb(perturbed, cell_emb_orig)

    for k in out_full_pert:
        if out_full_pert[k] is None:
            assert out_cached_pert[k] is None
            continue
        assert torch.equal(out_cached_pert[k], out_full_pert[k]), (
            f"Key {k} differs after perturbation"
        )
