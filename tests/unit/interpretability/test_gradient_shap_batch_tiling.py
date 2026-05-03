"""Unit tests for ``_BatchTilingWrapper._tile_fixed`` (gradient_shap_smoothgrad_attribution.py).

Captum's GradientSHAP / NoiseTunnel internally replicate ``pseudobulk``
along the batch dimension (n_samples copies). The encoder strictly
validates that ``cell_type_mask``, ``cell_offsets``, ``pathology`` etc.
share the same batch size as ``pseudobulk``. ``_BatchTilingWrapper``
detects the size mismatch and tiles the cached fixed inputs along dim 0
to match.

These tests exercise the synthetic-batch tiling-fidelity invariants
called out as a deferred follow-up in the Fix C summary (File 16).

Specifically we verify:
  * tile_factor == 1 returns the original kwargs unchanged.
  * cell_type_mask + pathology + per-sample tensors get
    ``repeat_interleave``-d along dim 0.
  * ccc_* graph edges pass through unchanged (graph-level, not batch-indexed).
  * cell_data / cell_offsets layout matches the per-sample-K-tile spec:
    sample i's K replicas occupy
    ``[K * cum_count[<i], K * cum_count[<i] + K * count_i)`` and
    ``cell_offsets_tiled[i*K + k, t] == sample_starts_tiled[i] + k * count_i + within_sample[i, t]``.
"""
from __future__ import annotations

import torch

from scripts.resdec_mhe.interpretability.gradient_shap_smoothgrad_attribution import (
    _BatchTilingWrapper,
)


class _StubBaseWrapper(torch.nn.Module):
    """Minimal stand-in for ``_ResDecCompositeWrapper`` that exposes
    ``_fixed_kwargs`` and a no-op ``set_fixed_inputs``."""

    def __init__(self):
        super().__init__()
        self._fixed_kwargs: dict = {}

    def set_fixed_inputs(self, batch: dict) -> None:
        # Mimic the canonical wrapper: cache the non-pseudobulk inputs.
        self._fixed_kwargs = {
            "ccc_edge_index": batch.get("ccc_edge_index"),
            "ccc_edge_type": batch.get("ccc_edge_type"),
            "ccc_edge_attr": batch.get("ccc_edge_attr"),
            "cell_type_mask": batch.get("cell_type_mask"),
            "pathology": batch.get("pathology"),
            "cell_data": batch.get("cell_data"),
            "cell_offsets": batch.get("cell_offsets"),
        }

    def forward(self, pseudobulk: torch.Tensor) -> torch.Tensor:
        # Stub — never invoked by these tests; we test _tile_fixed directly.
        return pseudobulk.sum(dim=(1, 2))


def _make_synthetic_batch(B: int = 2, n_types: int = 3, n_genes: int = 4):
    """Build a synthetic ragged-cells batch.

    Sample 0 has 4 cells (split across 3 types: 2/1/1).
    Sample 1 has 3 cells (split: 1/1/1).
    cell_data is dense [total_cells, n_genes] flat tensor.
    """
    # Per-sample cell counts (sum over types per sample).
    counts_per_sample = [4, 3]
    total = sum(counts_per_sample)
    cell_data = torch.arange(total * n_genes, dtype=torch.float32).reshape(total, n_genes)

    # cell_offsets: [B, n_types+1] with absolute pointers.
    # Sample 0: [0, 2, 3, 4]   (types 0/1/2 own cells [0,2)/[2,3)/[3,4))
    # Sample 1: [4, 5, 6, 7]   (types 0/1/2 own cells [4,5)/[5,6)/[6,7))
    cell_offsets = torch.tensor(
        [
            [0, 2, 3, 4],
            [4, 5, 6, 7],
        ],
        dtype=torch.int64,
    )

    cell_type_mask = torch.ones(B, n_types, dtype=torch.bool)
    pathology = torch.randn(B, 5)
    pseudobulk = torch.randn(B, n_types, n_genes)

    # Graph-level CCC edges (batch-invariant — should pass through unchanged).
    ccc_edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.int64)
    ccc_edge_type = torch.zeros(3, dtype=torch.int64)
    ccc_edge_attr = torch.randn(3, 8)

    return {
        "pseudobulk": pseudobulk,
        "cell_data": cell_data,
        "cell_offsets": cell_offsets,
        "cell_type_mask": cell_type_mask,
        "pathology": pathology,
        "ccc_edge_index": ccc_edge_index,
        "ccc_edge_type": ccc_edge_type,
        "ccc_edge_attr": ccc_edge_attr,
    }


class TestTileFixedFactorOne:
    """tile_factor == 1: identity-like behaviour."""

    def test_returns_copy_of_original_kwargs(self):
        base = _StubBaseWrapper()
        wrapper = _BatchTilingWrapper(base)
        batch = _make_synthetic_batch()
        wrapper.set_fixed_inputs(batch)
        out = wrapper._tile_fixed(1)
        assert set(out.keys()) == set(wrapper._original_fixed.keys())
        # Each value identical to the original.
        for k, v in out.items():
            orig = wrapper._original_fixed[k]
            if torch.is_tensor(v):
                torch.testing.assert_close(v, orig)
            else:
                assert v is orig


class TestTileFixedSimpleTensors:
    """Per-sample tensors (cell_type_mask, pathology) tile via repeat_interleave."""

    def test_cell_type_mask_repeat_interleaved(self):
        base = _StubBaseWrapper()
        wrapper = _BatchTilingWrapper(base)
        batch = _make_synthetic_batch(B=2, n_types=3)
        wrapper.set_fixed_inputs(batch)
        K = 5
        out = wrapper._tile_fixed(K)
        expected = batch["cell_type_mask"].repeat_interleave(K, dim=0)
        torch.testing.assert_close(out["cell_type_mask"], expected)

    def test_pathology_repeat_interleaved(self):
        base = _StubBaseWrapper()
        wrapper = _BatchTilingWrapper(base)
        batch = _make_synthetic_batch(B=2)
        wrapper.set_fixed_inputs(batch)
        K = 3
        out = wrapper._tile_fixed(K)
        expected = batch["pathology"].repeat_interleave(K, dim=0)
        torch.testing.assert_close(out["pathology"], expected)


class TestTileFixedCccEdges:
    """CCC graph edges (ccc_*) are batch-invariant — should pass through untouched."""

    def test_ccc_edges_pass_through(self):
        base = _StubBaseWrapper()
        wrapper = _BatchTilingWrapper(base)
        batch = _make_synthetic_batch()
        wrapper.set_fixed_inputs(batch)
        out = wrapper._tile_fixed(4)
        for key in ("ccc_edge_index", "ccc_edge_type", "ccc_edge_attr"):
            torch.testing.assert_close(out[key], batch[key])
            # Identity check — the original tensor object is reused.
            assert out[key] is batch[key]


class TestTileFixedCellDataLayout:
    """cell_data + cell_offsets must satisfy the per-sample-K-tile invariants."""

    def test_cell_data_total_size(self):
        """Tiled cell_data length == K * sum(per_sample_counts)."""
        base = _StubBaseWrapper()
        wrapper = _BatchTilingWrapper(base)
        batch = _make_synthetic_batch()
        wrapper.set_fixed_inputs(batch)
        K = 3
        out = wrapper._tile_fixed(K)
        # Sample counts: 4, 3 -> total 7. Tiled total = K * 7 = 21.
        assert out["cell_data"].shape[0] == K * 7
        assert out["cell_data"].shape[1] == 4  # n_genes preserved

    def test_cell_data_block_layout(self):
        """Sample i's K replicas of its cell-block lie consecutively in cell_data_tiled."""
        base = _StubBaseWrapper()
        wrapper = _BatchTilingWrapper(base)
        batch = _make_synthetic_batch()
        wrapper.set_fixed_inputs(batch)
        K = 2
        out = wrapper._tile_fixed(K)
        cell_data = batch["cell_data"]
        cell_offsets = batch["cell_offsets"]
        cd_tiled = out["cell_data"]
        # Sample 0: original [0:4), tiled occupies [0:8) (K=2 copies).
        s0, e0 = int(cell_offsets[0, 0]), int(cell_offsets[0, -1])
        block0 = cell_data[s0:e0]
        torch.testing.assert_close(cd_tiled[0:4], block0)
        torch.testing.assert_close(cd_tiled[4:8], block0)
        # Sample 1: original [4:7), tiled occupies [8:14).
        s1, e1 = int(cell_offsets[1, 0]), int(cell_offsets[1, -1])
        block1 = cell_data[s1:e1]
        torch.testing.assert_close(cd_tiled[8:11], block1)
        torch.testing.assert_close(cd_tiled[11:14], block1)

    def test_cell_offsets_pointer_invariant(self):
        """cell_offsets_tiled[i*K + k, t] - cell_offsets_tiled[i*K + k, 0]
        == within_sample[i, t] (shifts cancel)."""
        base = _StubBaseWrapper()
        wrapper = _BatchTilingWrapper(base)
        batch = _make_synthetic_batch()
        wrapper.set_fixed_inputs(batch)
        K = 3
        out = wrapper._tile_fixed(K)
        cell_offsets = batch["cell_offsets"]
        co_tiled = out["cell_offsets"]
        B0, n_types_plus_1 = cell_offsets.shape
        # Reconstruct within_sample from original.
        within_sample = cell_offsets - cell_offsets[:, :1]  # [B0, n_types+1]
        for i in range(B0):
            for k in range(K):
                row = co_tiled[i * K + k]
                # Shift cancels: row - row[0] == within_sample[i].
                torch.testing.assert_close(
                    row - row[0],
                    within_sample[i],
                )

    def test_cell_offsets_starts_match_cumsum(self):
        """For each sample i and replica k, the first offset
        ``co_tiled[i*K + k, 0]`` equals
        ``K * sum(counts_<i) + k * count_i``."""
        base = _StubBaseWrapper()
        wrapper = _BatchTilingWrapper(base)
        batch = _make_synthetic_batch()
        wrapper.set_fixed_inputs(batch)
        K = 2
        out = wrapper._tile_fixed(K)
        cell_offsets = batch["cell_offsets"]
        co_tiled = out["cell_offsets"]
        # counts: sample 0 -> 4, sample 1 -> 3.
        counts = (cell_offsets[:, -1] - cell_offsets[:, 0]).tolist()
        # Cum starts after tiling.
        cum_starts = [0]
        for c in counts[:-1]:
            cum_starts.append(K * c + cum_starts[-1])
        for i in range(len(counts)):
            for k in range(K):
                expected_start = cum_starts[i] + k * counts[i]
                assert int(co_tiled[i * K + k, 0]) == expected_start

    def test_cell_offsets_end_pointer_within_tiled_data(self):
        """The last offset of every tiled row must point inside ``cd_tiled``."""
        base = _StubBaseWrapper()
        wrapper = _BatchTilingWrapper(base)
        batch = _make_synthetic_batch()
        wrapper.set_fixed_inputs(batch)
        K = 4
        out = wrapper._tile_fixed(K)
        co_tiled = out["cell_offsets"]
        cd_tiled = out["cell_data"]
        assert int(co_tiled[:, -1].max()) <= cd_tiled.shape[0]


class TestTileFixedNoneTensors:
    """``None`` entries (missing kwargs) propagate without raising."""

    def test_none_kwarg_passes_through(self):
        base = _StubBaseWrapper()
        wrapper = _BatchTilingWrapper(base)
        batch = _make_synthetic_batch()
        # Drop cell_data + cell_offsets so the special branch sees None.
        batch["cell_data"] = None
        batch["cell_offsets"] = None
        wrapper.set_fixed_inputs(batch)
        out = wrapper._tile_fixed(2)
        # Both keys exist but values are None.
        assert out["cell_data"] is None
        assert out["cell_offsets"] is None
        # Other tensors still tile.
        torch.testing.assert_close(
            out["cell_type_mask"],
            batch["cell_type_mask"].repeat_interleave(2, dim=0),
        )
