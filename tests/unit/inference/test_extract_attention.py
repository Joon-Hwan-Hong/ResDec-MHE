"""
Tests for src/inference/extract_attention.py.

Test coverage includes:
- aggregate_hgt_attention() core behavior
- Edge type ordering (data-driven deterministic sort)
- Per-sample summaries
- Torch tensor handling
- Edge cases (empty input, missing edge types)

Integration-level HDF5 roundtrip tests live in tests/integration/test_hdf5_schema.py.
"""

import numpy as np
import pytest
import torch

from src.inference.extract_attention import aggregate_hgt_attention


# ============================================================================
# Realistic PyG-style edge type tuples for testing
# ============================================================================

# These mirror real HGT edge types: (src_cell_type, interaction_category, dst_cell_type)
EDGE_TYPE_A = ("Astrocyte", "Secreted_Signaling", "Microglia")
EDGE_TYPE_B = ("Microglia", "Cell_Cell_Contact", "Astrocyte")
EDGE_TYPE_C = ("Oligodendrocyte", "ECM_Receptor", "OPC")

DEFAULT_EDGE_TYPES = [EDGE_TYPE_A, EDGE_TYPE_B, EDGE_TYPE_C]


# ============================================================================
# Fixtures
# ============================================================================


def _make_hgt_attention(
    n_samples: int = 3,
    n_layers: int = 2,
    n_heads: int = 4,
    n_edges_per_type: int = 10,
    edge_types: list[tuple[str, str, str]] | None = None,
    use_torch: bool = False,
) -> list[list[dict]]:
    """Build synthetic HGT attention data with realistic tuple edge type keys."""
    if edge_types is None:
        edge_types = DEFAULT_EDGE_TYPES

    rng = np.random.RandomState(42)
    samples = []
    for _ in range(n_samples):
        layers = []
        for _ in range(n_layers):
            layer_dict = {}
            for et in edge_types:
                arr = rng.rand(n_edges_per_type, n_heads).astype(np.float32)
                if use_torch:
                    arr = torch.from_numpy(arr)
                layer_dict[et] = arr
            layers.append(layer_dict)
        samples.append(layers)
    return samples


@pytest.fixture
def hgt_attention():
    """Standard HGT attention with 3 edge types, 3 samples, 2 layers, 4 heads."""
    return _make_hgt_attention()


@pytest.fixture
def hgt_attention_torch():
    """HGT attention with torch tensors."""
    return _make_hgt_attention(use_torch=True)


# ============================================================================
# Core Behavior
# ============================================================================


class TestAggregateHGTAttention:
    """Tests for aggregate_hgt_attention function."""

    def test_returns_expected_keys(self, hgt_attention):
        """Result dict contains all expected keys."""
        result = aggregate_hgt_attention(hgt_attention)
        expected_keys = {
            "edge_type_names",
            "mean_by_edge_type",
            "std_by_edge_type",
            "per_sample",
            "n_samples",
            "n_layers",
            "n_samples_per_edge_type",
        }
        assert set(result.keys()) == expected_keys

    def test_output_shapes(self, hgt_attention):
        """Output arrays have correct shapes."""
        result = aggregate_hgt_attention(hgt_attention)

        n_edge_types = 3  # 3 edge types in fixture
        n_heads = 4

        assert result["mean_by_edge_type"].shape == (n_edge_types, n_heads)
        assert result["std_by_edge_type"].shape == (n_edge_types, n_heads)
        assert result["n_samples"] == 3
        assert result["n_layers"] == 2

    def test_per_sample_shape(self, hgt_attention):
        """Per-sample array has shape [n_samples, n_edge_types, n_layers, n_heads]."""
        result = aggregate_hgt_attention(hgt_attention, include_per_sample=True)
        per_sample = result["per_sample"]

        assert per_sample.shape == (3, 3, 2, 4)

    def test_per_sample_disabled(self, hgt_attention):
        """Per-sample is None when disabled."""
        result = aggregate_hgt_attention(hgt_attention, include_per_sample=False)
        assert result["per_sample"] is None

    def test_mean_std_consistent_with_per_sample(self, hgt_attention):
        """Mean and std match manual computation from per-sample data."""
        result = aggregate_hgt_attention(hgt_attention, include_per_sample=True)

        per_sample = result["per_sample"]
        # Mean across layers -> [n_samples, n_edge_types, n_heads]
        attention_per_sample = per_sample.mean(axis=2)
        expected_mean = attention_per_sample.mean(axis=0)
        expected_std = attention_per_sample.std(axis=0)

        np.testing.assert_allclose(result["mean_by_edge_type"], expected_mean, atol=1e-6)
        np.testing.assert_allclose(result["std_by_edge_type"], expected_std, atol=1e-6)

    def test_attention_values_are_edge_means(self, hgt_attention):
        """Per-sample values are means across edges (not sums or raw values)."""
        result = aggregate_hgt_attention(hgt_attention, include_per_sample=True)

        # Manually compute expected for first sample, first discovered edge type, first layer
        # Edge types are sorted by str(), so find sorted order
        sorted_types = sorted(DEFAULT_EDGE_TYPES, key=str)
        first_et = sorted_types[0]
        raw_attn = hgt_attention[0][0][first_et]  # [n_edges, n_heads]
        expected = raw_attn.mean(axis=0)  # [n_heads]

        actual = result["per_sample"][0, 0, 0, :]  # first sample, first edge type, first layer

        np.testing.assert_allclose(actual, expected, atol=1e-6)

    def test_edge_type_names_are_pipe_separated_tuples(self, hgt_attention):
        """Edge type names are formatted as 'src|interaction|dst'."""
        result = aggregate_hgt_attention(hgt_attention)

        for name in result["edge_type_names"]:
            parts = name.split("|")
            assert len(parts) == 3, f"Expected 3 parts in '{name}', got {len(parts)}"
            # Each part should be a real cell type or interaction name, not single chars
            assert all(len(p) > 1 for p in parts), f"Got single-char parts in '{name}'"


# ============================================================================
# Edge Type Ordering
# ============================================================================


class TestEdgeTypeOrdering:
    """Tests for data-driven edge type ordering with deterministic sort."""

    def test_discovers_edge_types_from_data(self, hgt_attention):
        """When edge_types=None, discovers all types from data."""
        result = aggregate_hgt_attention(hgt_attention)

        # Should find all 3 edge types
        assert len(result["edge_type_names"]) == 3

    def test_deterministic_sort_order(self):
        """Discovery produces deterministic sorted ordering across runs."""
        # Create data with edge types in different insertion order
        et_forward = [EDGE_TYPE_A, EDGE_TYPE_B, EDGE_TYPE_C]
        et_reverse = [EDGE_TYPE_C, EDGE_TYPE_B, EDGE_TYPE_A]

        data_forward = _make_hgt_attention(edge_types=et_forward)
        data_reverse = _make_hgt_attention(edge_types=et_reverse)

        result_forward = aggregate_hgt_attention(data_forward)
        result_reverse = aggregate_hgt_attention(data_reverse)

        # Same edge types in same order regardless of data insertion order
        assert result_forward["edge_type_names"] == result_reverse["edge_type_names"]

    def test_custom_edge_types_respected(self, hgt_attention):
        """Custom edge_types parameter overrides data-driven discovery."""
        custom_types = [EDGE_TYPE_A, EDGE_TYPE_B]
        result = aggregate_hgt_attention(hgt_attention, edge_types=custom_types)

        expected_names = [f"{et[0]}|{et[1]}|{et[2]}" for et in custom_types]
        assert result["edge_type_names"] == expected_names
        assert result["mean_by_edge_type"].shape[0] == 2

    def test_missing_edge_types_get_nan(self):
        """Edge types requested but not in data get NaN attention (not zero)."""
        # Create data with only one edge type
        data = _make_hgt_attention(n_samples=2, n_layers=1, edge_types=[EDGE_TYPE_A])

        # But request all 3 edge types explicitly
        result = aggregate_hgt_attention(data, edge_types=DEFAULT_EDGE_TYPES)

        # EDGE_TYPE_A should have non-NaN attention
        sorted_ets = sorted(DEFAULT_EDGE_TYPES, key=str)
        et_a_idx = sorted_ets.index(EDGE_TYPE_A)
        assert not np.any(np.isnan(result["per_sample"][:, et_a_idx, :, :]))

        # Others should be NaN (absent, not zero-biased)
        for idx, et in enumerate(sorted_ets):
            if et != EDGE_TYPE_A:
                assert np.all(np.isnan(result["per_sample"][:, idx, :, :]))

    def test_only_data_edge_types_when_no_explicit_list(self):
        """Without explicit edge_types, only data-present types appear in output."""
        # Create data with 1 edge type
        data = _make_hgt_attention(n_samples=2, n_layers=1, edge_types=[EDGE_TYPE_B])
        result = aggregate_hgt_attention(data)

        assert len(result["edge_type_names"]) == 1
        assert result["edge_type_names"][0] == f"{EDGE_TYPE_B[0]}|{EDGE_TYPE_B[1]}|{EDGE_TYPE_B[2]}"


# ============================================================================
# Torch Tensor Handling
# ============================================================================


class TestTorchTensorHandling:
    """Tests for torch.Tensor → numpy conversion."""

    def test_torch_tensors_converted(self, hgt_attention_torch):
        """Torch tensors are correctly converted to numpy in output."""
        result = aggregate_hgt_attention(hgt_attention_torch)

        assert isinstance(result["mean_by_edge_type"], np.ndarray)
        assert isinstance(result["std_by_edge_type"], np.ndarray)
        assert isinstance(result["per_sample"], np.ndarray)

    def test_torch_and_numpy_produce_same_results(self, hgt_attention, hgt_attention_torch):
        """Torch and numpy inputs produce identical results."""
        result_np = aggregate_hgt_attention(hgt_attention)
        result_torch = aggregate_hgt_attention(hgt_attention_torch)

        np.testing.assert_allclose(
            result_np["mean_by_edge_type"],
            result_torch["mean_by_edge_type"],
            atol=1e-6,
        )
        np.testing.assert_allclose(
            result_np["per_sample"],
            result_torch["per_sample"],
            atol=1e-6,
        )


# ============================================================================
# Edge Cases
# ============================================================================


class TestEdgeCases:
    """Edge case tests for boundary conditions."""

    def test_empty_input(self):
        """Empty list returns empty result with zero counts."""
        result = aggregate_hgt_attention([])
        assert result["edge_type_names"] == []
        assert result["n_samples"] == 0
        assert result["n_layers"] == 0
        assert result["mean_by_edge_type"].size == 0

    def test_empty_edge_types(self, hgt_attention):
        """Empty edge_types list returns empty arrays."""
        result = aggregate_hgt_attention(hgt_attention, edge_types=[])
        assert result["edge_type_names"] == []
        assert result["mean_by_edge_type"].size == 0
        assert result["n_samples"] == 3
        assert result["n_layers"] == 2

    def test_single_sample(self):
        """Single sample produces zero std."""
        data = _make_hgt_attention(n_samples=1)
        result = aggregate_hgt_attention(data)
        np.testing.assert_array_equal(result["std_by_edge_type"], 0.0)

    def test_single_layer(self):
        """Single layer works correctly."""
        data = _make_hgt_attention(n_layers=1)
        result = aggregate_hgt_attention(data, include_per_sample=True)
        assert result["n_layers"] == 1
        assert result["per_sample"].shape[2] == 1

    def test_single_head(self):
        """Single head works correctly."""
        data = _make_hgt_attention(n_heads=1)
        result = aggregate_hgt_attention(data)
        assert result["mean_by_edge_type"].shape[-1] == 1


# ============================================================================
# Phase 6 Review Round 8 — H3: NaN-based absent edge type handling
# ============================================================================


class TestAbsentEdgeTypeNaN:
    """Tests that absent edge types use NaN, not zero, to avoid biasing means."""

    def test_absent_edge_type_is_nan_not_zero(self):
        """Sample missing an edge type should have NaN in per_sample array."""
        n_samples, n_layers, n_heads = 3, 2, 4
        # Build attention where sample 0 is missing EDGE_TYPE_C
        data = _make_hgt_attention(
            n_samples=n_samples, n_layers=n_layers, n_heads=n_heads
        )
        # Remove EDGE_TYPE_C from sample 0's layers
        for layer_attn in data[0]:
            if EDGE_TYPE_C in layer_attn:
                del layer_attn[EDGE_TYPE_C]

        result = aggregate_hgt_attention(data, include_per_sample=True)
        per_sample = result["per_sample"]
        # Find index of EDGE_TYPE_C
        sorted_ets = sorted(DEFAULT_EDGE_TYPES, key=str)
        c_idx = sorted_ets.index(EDGE_TYPE_C)
        # Sample 0 should be NaN for EDGE_TYPE_C
        assert np.all(np.isnan(per_sample[0, c_idx, :, :]))
        # Samples 1,2 should NOT be NaN for EDGE_TYPE_C
        assert not np.any(np.isnan(per_sample[1, c_idx, :, :]))
        assert not np.any(np.isnan(per_sample[2, c_idx, :, :]))

    def test_sparse_edge_type_mean_excludes_absent(self):
        """Edge type present in 2/5 samples → mean of 2 samples, not diluted."""
        n_samples, n_layers, n_heads = 5, 1, 2
        rng = np.random.RandomState(42)
        # Only samples 0 and 1 have EDGE_TYPE_A
        data = []
        for s in range(n_samples):
            layer_attn = {}
            for et in DEFAULT_EDGE_TYPES:
                if et == EDGE_TYPE_A and s >= 2:
                    continue  # absent for samples 2,3,4
                layer_attn[et] = rng.rand(10, n_heads)
            data.append([layer_attn])

        result = aggregate_hgt_attention(data, include_per_sample=True)
        sorted_ets = sorted(DEFAULT_EDGE_TYPES, key=str)
        a_idx = sorted_ets.index(EDGE_TYPE_A)

        # Mean should be computed from only 2 samples, not 5
        per_sample = result["per_sample"]
        valid_vals = per_sample[:, a_idx, 0, :]  # [n_samples, n_heads]
        expected_mean = np.nanmean(valid_vals, axis=0)
        np.testing.assert_array_almost_equal(
            result["mean_by_edge_type"][a_idx], expected_mean
        )

    def test_n_samples_per_edge_type_correct(self):
        """n_samples_per_edge_type reflects actual edge type presence."""
        n_samples = 5
        data = _make_hgt_attention(n_samples=n_samples, n_layers=1)
        # Remove EDGE_TYPE_B from samples 3 and 4
        sorted_ets = sorted(DEFAULT_EDGE_TYPES, key=str)
        for s in [3, 4]:
            if EDGE_TYPE_B in data[s][0]:
                del data[s][0][EDGE_TYPE_B]

        result = aggregate_hgt_attention(data)
        b_idx = sorted_ets.index(EDGE_TYPE_B)
        assert result["n_samples_per_edge_type"][b_idx] == 3  # 5 - 2 removed
