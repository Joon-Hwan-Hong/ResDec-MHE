"""
Stress and Scale Tests for Model Components.

Tests verifying model behavior at scale:
1. Batch Size Scaling - Various batch sizes work correctly
2. Cell Count Scaling - Many cells per cell type
3. Gene Count Scaling - Various gene counts (1000-5000)
4. Edge Index Scaling - Sparse to fully connected edges
5. Numerical Stability at Scale - No NaN/overflow at large scales
6. Memory Efficiency - Memory scales reasonably

Usage:
    # Run all stress tests
    pytest tests/unit/models/test_stress_scale.py -v

    # Run only slow tests (marked with @pytest.mark.slow)
    pytest tests/unit/models/test_stress_scale.py -v -m slow

    # Skip slow tests
    pytest tests/unit/models/test_stress_scale.py -v -m "not slow"
"""

import gc
import sys
from typing import Optional

import pytest
import torch
import torch.nn as nn

from src.data.constants import N_CELL_TYPES, N_REGIONS, CELL_TYPE_ORDER, ALL_EDGE_TYPES


# ============================================================================
# Memory Monitoring Utilities
# ============================================================================


def get_memory_usage() -> dict[str, float]:
    """Get current memory usage in MB."""
    result = {'cpu_mb': 0.0, 'gpu_mb': 0.0, 'gpu_allocated_mb': 0.0}

    # CPU memory (approximate via sys.getsizeof is not accurate, skip for now)
    # For accurate CPU measurement, would need psutil

    # GPU memory
    if torch.cuda.is_available():
        result['gpu_mb'] = torch.cuda.memory_reserved() / (1024 * 1024)
        result['gpu_allocated_mb'] = torch.cuda.memory_allocated() / (1024 * 1024)

    return result


def clear_memory():
    """Clear CUDA cache and run garbage collection."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def check_memory_available(required_mb: float = 1000.0) -> bool:
    """Check if enough GPU memory is available."""
    if not torch.cuda.is_available():
        return True  # CPU tests always proceed

    free_memory = torch.cuda.get_device_properties(0).total_memory
    allocated = torch.cuda.memory_allocated()
    available_mb = (free_memory - allocated) / (1024 * 1024)

    return available_mb >= required_mb


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def device():
    """Get available device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def cpu_device():
    """Force CPU device."""
    return torch.device("cpu")


@pytest.fixture
def mini_node_types():
    """Small set of node types for faster tests."""
    return ["Astrocyte", "Oligodendrocyte", "Microglia", "CGE interneuron"]


@pytest.fixture
def mini_edge_categories():
    """Small set of edge categories for faster tests."""
    return ["Secreted_Signaling", "ECM_Receptor"]


@pytest.fixture
def small_model_config():
    """Small model configuration for stress tests."""
    return {
        'n_genes': 100,
        'n_cell_types': N_CELL_TYPES,
        'd_embed': 32,
        'd_fused': 32,
        'd_cond': 16,
        'n_regions': N_REGIONS,
        'n_hgt_layers': 1,
        'n_hgt_heads': 4,
        'n_isab_layers': 1,
        'n_inducing_points': 4,
        'n_attention_heads': 4,
        'd_head_hidden': 16,
        'dropout': 0.0,  # Disable for deterministic testing
    }


def create_sample_inputs(
    batch_size: int,
    n_genes: int = 100,
    n_cell_types: int = N_CELL_TYPES,
    max_cells: int = 10,
    n_edges: int = 20,
    n_regions: int = N_REGIONS,
    device: torch.device = torch.device("cpu"),
) -> dict:
    """Create sample inputs for the full model."""
    from src.data.constants import CELL_TYPE_ORDER, ALL_EDGE_TYPES, sanitize_key

    # Build edge_index_dict_list and edge_attr_dict_list
    edge_index_dict_list = []
    edge_attr_dict_list = []
    for _ in range(batch_size):
        edge_index_dict = {}
        edge_attr_dict = {}
        for src_ct in CELL_TYPE_ORDER[:3]:
            for dst_ct in CELL_TYPE_ORDER[:3]:
                for et in ALL_EDGE_TYPES[:2]:
                    key = (sanitize_key(src_ct), sanitize_key(et), sanitize_key(dst_ct))
                    edge_index_dict[key] = torch.zeros(2, n_edges, dtype=torch.long, device=device)
                    edge_attr_dict[key] = torch.rand(n_edges, 1, device=device)
        edge_index_dict_list.append(edge_index_dict)
        edge_attr_dict_list.append(edge_attr_dict)

    return {
        'region_pseudobulk': torch.randn(
            batch_size, n_regions, n_cell_types, n_genes, device=device
        ),
        'region_mask': torch.ones(
            batch_size, n_regions, dtype=torch.bool, device=device
        ),
        'edge_index_dict_list': edge_index_dict_list,
        'edge_attr_dict_list': edge_attr_dict_list,
        'cells': torch.randn(
            batch_size, n_cell_types, max_cells, n_genes, device=device
        ),
        'cell_mask': torch.ones(
            batch_size, n_cell_types, max_cells, dtype=torch.bool, device=device
        ),
        'pathology': torch.randn(batch_size, 3, device=device),
        'cognition': torch.randn(batch_size, 1, device=device),
    }


# ============================================================================
# Batch Size Scaling Tests
# ============================================================================


class TestBatchSizeScaling:
    """Test model behavior with various batch sizes."""

    @pytest.mark.parametrize("batch_size", [1, 16])
    def test_batch_size_cpu(self, small_model_config, cpu_device, batch_size):
        """Batch size of {batch_size} works correctly on CPU."""
        from src.models.full_model import CognitiveResilienceModel

        model = CognitiveResilienceModel(**small_model_config, use_bayesian_head=False)
        model = model.to(cpu_device)
        model.eval()

        inputs = create_sample_inputs(
            batch_size=batch_size,
            n_genes=small_model_config['n_genes'],
            device=cpu_device,
        )

        with torch.no_grad():
            output = model(**inputs)

        assert output['mean'].shape == (batch_size, 1)
        assert not torch.isnan(output['mean']).any()
        assert not torch.isinf(output['mean']).any()

    @pytest.mark.slow
    def test_batch_size_64(self, small_model_config, device):
        """Batch size of 64 works correctly."""
        from src.models.full_model import CognitiveResilienceModel

        # Skip if insufficient memory
        if device.type == 'cuda' and not check_memory_available(2000):
            pytest.skip("Insufficient GPU memory for batch_size=64")

        model = CognitiveResilienceModel(**small_model_config, use_bayesian_head=False)
        model = model.to(device)
        model.eval()

        inputs = create_sample_inputs(
            batch_size=64,
            n_genes=small_model_config['n_genes'],
            device=device,
        )

        with torch.no_grad():
            output = model(**inputs)

        assert output['mean'].shape == (64, 1)
        assert not torch.isnan(output['mean']).any()
        assert not torch.isinf(output['mean']).any()

        clear_memory()


# ============================================================================
# Cell Count Scaling Tests
# ============================================================================


class TestCellCountScaling:
    """Test model behavior with varying cell counts per cell type."""

    def test_max_cells_100(self, small_model_config, cpu_device):
        """100 cells per cell type works correctly."""
        from src.models.branches.cell_transformer import CellTransformer

        transformer = CellTransformer(
            n_genes=small_model_config['n_genes'],
            n_cell_types=small_model_config['n_cell_types'],
            d_model=small_model_config['d_embed'],
            n_heads=small_model_config['n_hgt_heads'],
            n_isab_layers=small_model_config['n_isab_layers'],
            n_inducing=small_model_config['n_inducing_points'],
            dropout=0.0,
        ).to(cpu_device)
        transformer.eval()

        max_cells = 100
        B = 2
        cells = torch.randn(
            B, small_model_config['n_cell_types'], max_cells,
            small_model_config['n_genes'], device=cpu_device
        )
        cell_mask = torch.ones(
            B, small_model_config['n_cell_types'], max_cells,
            dtype=torch.bool, device=cpu_device
        )

        with torch.no_grad():
            output, selection_weights, _ = transformer(cells, cell_mask)

        assert output.shape == (B, small_model_config['n_cell_types'], small_model_config['d_embed'])
        assert not torch.isnan(output).any()

    @pytest.mark.slow
    def test_max_cells_500(self, small_model_config, cpu_device):
        """500 cells per cell type works correctly."""
        from src.models.branches.cell_transformer import CellTransformer

        transformer = CellTransformer(
            n_genes=small_model_config['n_genes'],
            n_cell_types=small_model_config['n_cell_types'],
            d_model=small_model_config['d_embed'],
            n_heads=small_model_config['n_hgt_heads'],
            n_isab_layers=small_model_config['n_isab_layers'],
            n_inducing=small_model_config['n_inducing_points'],
            dropout=0.0,
        ).to(cpu_device)
        transformer.eval()

        max_cells = 500
        B = 1  # Small batch due to memory
        cells = torch.randn(
            B, small_model_config['n_cell_types'], max_cells,
            small_model_config['n_genes'], device=cpu_device
        )
        cell_mask = torch.ones(
            B, small_model_config['n_cell_types'], max_cells,
            dtype=torch.bool, device=cpu_device
        )

        with torch.no_grad():
            output, selection_weights, _ = transformer(cells, cell_mask)

        assert output.shape == (B, small_model_config['n_cell_types'], small_model_config['d_embed'])
        assert not torch.isnan(output).any()

        clear_memory()

    @pytest.mark.slow
    def test_max_cells_1000(self, small_model_config, device):
        """1000 cells per cell type (expected max) works correctly."""
        from src.models.branches.cell_transformer import CellTransformer

        # Skip if insufficient memory
        if device.type == 'cuda' and not check_memory_available(4000):
            pytest.skip("Insufficient GPU memory for max_cells=1000")

        # Use smaller subset for memory efficiency
        n_cell_types = 4  # Subset

        transformer = CellTransformer(
            n_genes=small_model_config['n_genes'],
            n_cell_types=n_cell_types,
            d_model=small_model_config['d_embed'],
            n_heads=small_model_config['n_hgt_heads'],
            n_isab_layers=small_model_config['n_isab_layers'],
            n_inducing=small_model_config['n_inducing_points'],
            dropout=0.0,
        ).to(device)
        transformer.eval()

        max_cells = 1000
        B = 1
        cells = torch.randn(
            B, n_cell_types, max_cells,
            small_model_config['n_genes'], device=device
        )
        cell_mask = torch.ones(
            B, n_cell_types, max_cells,
            dtype=torch.bool, device=device
        )

        with torch.no_grad():
            output, selection_weights, _ = transformer(cells, cell_mask)

        assert output.shape == (B, n_cell_types, small_model_config['d_embed'])
        assert not torch.isnan(output).any()

        clear_memory()


# ============================================================================
# Gene Count Scaling Tests
# ============================================================================


class TestGeneCountScaling:
    """Test model behavior with varying gene counts."""

    @pytest.mark.parametrize("n_genes", [1000, 3000])
    def test_n_genes_cpu(self, cpu_device, n_genes):
        """PseudobulkEncoder works correctly with {n_genes} genes on CPU."""
        from src.models.branches.pseudobulk_encoder import PseudobulkEncoder

        n_cell_types = N_CELL_TYPES
        d_embed = 64

        encoder = PseudobulkEncoder(
            n_cell_types=n_cell_types,
            n_genes=n_genes,
            d_embed=d_embed,
            dropout=0.0,
        ).to(cpu_device)
        encoder.eval()

        B = 4
        x = torch.randn(B, n_cell_types, n_genes, device=cpu_device)

        with torch.no_grad():
            output = encoder(x)

        assert output.shape == (B, n_cell_types, d_embed)
        assert not torch.isnan(output).any()

    @pytest.mark.slow
    def test_n_genes_5000(self, device):
        """5000 genes (stress) works correctly."""
        from src.models.branches.pseudobulk_encoder import PseudobulkEncoder

        n_genes = 5000
        n_cell_types = N_CELL_TYPES
        d_embed = 64

        encoder = PseudobulkEncoder(
            n_cell_types=n_cell_types,
            n_genes=n_genes,
            d_embed=d_embed,
            dropout=0.0,
        ).to(device)
        encoder.eval()

        B = 4
        x = torch.randn(B, n_cell_types, n_genes, device=device)

        with torch.no_grad():
            output = encoder(x)

        assert output.shape == (B, n_cell_types, d_embed)
        assert not torch.isnan(output).any()

        clear_memory()


# ============================================================================
# Edge Index Scaling Tests
# ============================================================================


class TestEdgeIndexScaling:
    """Test HGT encoder with varying edge densities."""

    def test_sparse_edge_index(self, mini_node_types, mini_edge_categories, cpu_device):
        """Very few edges (10) works correctly."""
        from src.models.branches.hgt_encoder import HGTEncoder

        d_input = 32
        encoder = HGTEncoder(
            d_input=d_input,
            d_hidden=32,
            d_output=32,
            n_heads=2,
            n_layers=1,
            dropout=0.0,
            edge_dim=1,
            node_types=mini_node_types,
            edge_categories=mini_edge_categories,
        ).to(cpu_device)
        encoder.eval()

        # Create very sparse graph - 10 edges
        x_dict = {nt: torch.randn(1, d_input, device=cpu_device) for nt in mini_node_types}

        # Only 2-3 edge connections
        edge_index_dict = {
            (mini_node_types[0], mini_edge_categories[0], mini_node_types[1]):
                torch.tensor([[0], [0]], device=cpu_device),
            (mini_node_types[1], mini_edge_categories[0], mini_node_types[2]):
                torch.tensor([[0], [0]], device=cpu_device),
        }
        edge_attr_dict = {
            (mini_node_types[0], mini_edge_categories[0], mini_node_types[1]):
                torch.rand(1, 1, device=cpu_device),
            (mini_node_types[1], mini_edge_categories[0], mini_node_types[2]):
                torch.rand(1, 1, device=cpu_device),
        }

        with torch.no_grad():
            output_dict, _ = encoder(x_dict, edge_index_dict, edge_attr_dict)

        for nt in mini_node_types:
            assert nt in output_dict
            assert not torch.isnan(output_dict[nt]).any()

    def test_dense_edge_index(self, mini_node_types, mini_edge_categories, cpu_device):
        """Many edges (1000+) works correctly."""
        from src.models.branches.hgt_encoder import HGTEncoder

        d_input = 32
        n_nodes_per_type = 10

        encoder = HGTEncoder(
            d_input=d_input,
            d_hidden=32,
            d_output=32,
            n_heads=2,
            n_layers=1,
            dropout=0.0,
            edge_dim=1,
            node_types=mini_node_types,
            edge_categories=mini_edge_categories,
        ).to(cpu_device)
        encoder.eval()

        # Create dense graph - many nodes, many edges
        x_dict = {
            nt: torch.randn(n_nodes_per_type, d_input, device=cpu_device)
            for nt in mini_node_types
        }

        # Create many edges between types
        edge_index_dict = {}
        edge_attr_dict = {}

        for src_type in mini_node_types:
            for dst_type in mini_node_types:
                for edge_cat in mini_edge_categories:
                    n_edges = 50  # 50 edges per type pair
                    edge_key = (src_type, edge_cat, dst_type)

                    src_idx = torch.randint(0, n_nodes_per_type, (n_edges,), device=cpu_device)
                    dst_idx = torch.randint(0, n_nodes_per_type, (n_edges,), device=cpu_device)

                    edge_index_dict[edge_key] = torch.stack([src_idx, dst_idx], dim=0)
                    edge_attr_dict[edge_key] = torch.rand(n_edges, 1, device=cpu_device)

        with torch.no_grad():
            output_dict, _ = encoder(x_dict, edge_index_dict, edge_attr_dict)

        for nt in mini_node_types:
            assert nt in output_dict
            assert output_dict[nt].shape == (n_nodes_per_type, 32)
            assert not torch.isnan(output_dict[nt]).any()

    def test_fully_connected_edges(self, mini_node_types, mini_edge_categories, cpu_device):
        """All possible edges (fully connected) works correctly."""
        from src.models.branches.hgt_encoder import HGTEncoder

        d_input = 32
        n_nodes_per_type = 5

        encoder = HGTEncoder(
            d_input=d_input,
            d_hidden=32,
            d_output=32,
            n_heads=2,
            n_layers=1,
            dropout=0.0,
            edge_dim=1,
            node_types=mini_node_types,
            edge_categories=mini_edge_categories,
        ).to(cpu_device)
        encoder.eval()

        # Create fully connected graph
        x_dict = {
            nt: torch.randn(n_nodes_per_type, d_input, device=cpu_device)
            for nt in mini_node_types
        }

        edge_index_dict = {}
        edge_attr_dict = {}

        for src_type in mini_node_types:
            for dst_type in mini_node_types:
                for edge_cat in mini_edge_categories:
                    edge_key = (src_type, edge_cat, dst_type)

                    # All-to-all edges
                    src_idx = []
                    dst_idx = []
                    for i in range(n_nodes_per_type):
                        for j in range(n_nodes_per_type):
                            src_idx.append(i)
                            dst_idx.append(j)

                    edge_index_dict[edge_key] = torch.tensor(
                        [src_idx, dst_idx], device=cpu_device
                    )
                    n_edges = len(src_idx)
                    edge_attr_dict[edge_key] = torch.rand(n_edges, 1, device=cpu_device)

        with torch.no_grad():
            output_dict, attn = encoder(
                x_dict, edge_index_dict, edge_attr_dict, return_attention=True
            )

        for nt in mini_node_types:
            assert nt in output_dict
            assert not torch.isnan(output_dict[nt]).any()
            assert not torch.isinf(output_dict[nt]).any()


# ============================================================================
# Numerical Stability at Scale Tests
# ============================================================================


class TestNumericalStabilityAtScale:
    """Test numerical stability with large inputs."""

    @pytest.mark.slow
    def test_large_batch_no_nan(self, small_model_config, device):
        """Large batch produces no NaN values."""
        from src.models.full_model import CognitiveResilienceModel

        if device.type == 'cuda' and not check_memory_available(2000):
            pytest.skip("Insufficient GPU memory")

        model = CognitiveResilienceModel(**small_model_config, use_bayesian_head=False)
        model = model.to(device)
        model.eval()

        batch_size = 32
        inputs = create_sample_inputs(
            batch_size=batch_size,
            n_genes=small_model_config['n_genes'],
            device=device,
        )

        with torch.no_grad():
            output = model(**inputs)

        assert not torch.isnan(output['mean']).any(), "NaN detected in output"
        assert not torch.isinf(output['mean']).any(), "Inf detected in output"
        assert not torch.isnan(output['attention_weights']).any(), "NaN in attention"

        clear_memory()

    @pytest.mark.slow
    def test_many_cells_no_overflow(self, small_model_config, device):
        """Many cells doesn't cause overflow."""
        from src.models.branches.cell_transformer import CellTransformer

        if device.type == 'cuda' and not check_memory_available(2000):
            pytest.skip("Insufficient GPU memory")

        transformer = CellTransformer(
            n_genes=small_model_config['n_genes'],
            n_cell_types=8,  # Subset for memory
            d_model=small_model_config['d_embed'],
            n_heads=small_model_config['n_hgt_heads'],
            n_isab_layers=small_model_config['n_isab_layers'],
            n_inducing=small_model_config['n_inducing_points'],
            dropout=0.0,
        ).to(device)
        transformer.eval()

        max_cells = 500
        B = 2
        n_cell_types = 8

        cells = torch.randn(
            B, n_cell_types, max_cells,
            small_model_config['n_genes'], device=device
        )
        cell_mask = torch.ones(
            B, n_cell_types, max_cells,
            dtype=torch.bool, device=device
        )

        with torch.no_grad():
            output, selection_weights, _ = transformer(cells, cell_mask)

        assert not torch.isnan(output).any(), "NaN in output"
        assert not torch.isinf(output).any(), "Inf in output"

        # Check output is in reasonable range
        assert output.abs().max() < 1000, "Output magnitude too large"

        clear_memory()

    def test_gradient_magnitude_reasonable(self, small_model_config, cpu_device):
        """Gradients stay reasonable at scale."""
        from src.models.full_model import CognitiveResilienceModel

        model = CognitiveResilienceModel(**small_model_config, use_bayesian_head=False)
        model = model.to(cpu_device)
        model.train()

        batch_size = 8
        inputs = create_sample_inputs(
            batch_size=batch_size,
            n_genes=small_model_config['n_genes'],
            device=cpu_device,
        )

        output = model(**inputs)
        loss = output['mean'].sum()
        loss.backward()

        # Check gradient magnitudes
        max_grad_norm = 0.0
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                max_grad_norm = max(max_grad_norm, grad_norm)

                # No gradient should be NaN or Inf
                assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"
                assert not torch.isinf(param.grad).any(), f"Inf gradient in {name}"

        # Maximum gradient norm should be reasonable (not exploding)
        assert max_grad_norm < 1e6, f"Gradient explosion: max_norm={max_grad_norm}"

    def test_extreme_input_values_handled(self, small_model_config, cpu_device):
        """Model handles extreme input values gracefully."""
        from src.models.branches.pseudobulk_encoder import PseudobulkEncoder

        encoder = PseudobulkEncoder(
            n_cell_types=small_model_config['n_cell_types'],
            n_genes=small_model_config['n_genes'],
            d_embed=small_model_config['d_embed'],
            dropout=0.0,
        ).to(cpu_device)
        encoder.eval()

        B = 2
        n_cell_types = small_model_config['n_cell_types']
        n_genes = small_model_config['n_genes']

        # Test with large values
        x_large = torch.randn(B, n_cell_types, n_genes, device=cpu_device) * 100
        with torch.no_grad():
            out_large = encoder(x_large)
        assert not torch.isnan(out_large).any(), "NaN with large inputs"

        # Test with small values
        x_small = torch.randn(B, n_cell_types, n_genes, device=cpu_device) * 1e-6
        with torch.no_grad():
            out_small = encoder(x_small)
        assert not torch.isnan(out_small).any(), "NaN with small inputs"

        # Test with zeros
        x_zeros = torch.zeros(B, n_cell_types, n_genes, device=cpu_device)
        with torch.no_grad():
            out_zeros = encoder(x_zeros)
        assert not torch.isnan(out_zeros).any(), "NaN with zero inputs"


# ============================================================================
# Memory Efficiency Tests
# ============================================================================


class TestMemoryEfficiency:
    """Test memory usage characteristics."""

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="Memory tests require CUDA for accurate measurement"
    )
    def test_memory_scales_linearly_with_batch(self, small_model_config):
        """Memory scales approximately linearly with batch size."""
        from src.models.full_model import CognitiveResilienceModel

        device = torch.device("cuda:0")

        model = CognitiveResilienceModel(**small_model_config, use_bayesian_head=False)
        model = model.to(device)
        model.eval()

        clear_memory()
        baseline = torch.cuda.memory_allocated(device)

        memory_readings = []
        batch_sizes = [1, 2, 4, 8]

        for batch_size in batch_sizes:
            clear_memory()

            inputs = create_sample_inputs(
                batch_size=batch_size,
                n_genes=small_model_config['n_genes'],
                device=device,
            )

            with torch.no_grad():
                output = model(**inputs)

            # Force synchronization
            torch.cuda.synchronize()

            peak_memory = torch.cuda.max_memory_allocated(device) - baseline
            memory_readings.append(peak_memory / (1024 * 1024))  # Convert to MB

            del inputs, output
            torch.cuda.reset_peak_memory_stats(device)

        # Check that memory roughly scales linearly
        # Memory for batch_size=8 should be roughly 8x memory for batch_size=1
        # Allow significant tolerance (factor of 3) due to fixed overhead
        ratio = memory_readings[-1] / max(memory_readings[0], 0.1)  # Avoid div by 0

        assert ratio < batch_sizes[-1] * 3, (
            f"Memory scaling not linear: batch 1={memory_readings[0]:.1f}MB, "
            f"batch 8={memory_readings[-1]:.1f}MB, ratio={ratio:.1f}"
        )

        clear_memory()

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="Memory tests require CUDA for accurate measurement"
    )
    def test_memory_with_large_cells(self, small_model_config):
        """Memory stays bounded with large cell counts."""
        from src.models.branches.cell_transformer import CellTransformer

        device = torch.device("cuda:0")

        # Skip if insufficient memory
        if not check_memory_available(4000):
            pytest.skip("Insufficient GPU memory")

        transformer = CellTransformer(
            n_genes=small_model_config['n_genes'],
            n_cell_types=4,  # Small subset
            d_model=small_model_config['d_embed'],
            n_heads=small_model_config['n_hgt_heads'],
            n_isab_layers=small_model_config['n_isab_layers'],
            n_inducing=small_model_config['n_inducing_points'],
            dropout=0.0,
        ).to(device)
        transformer.eval()

        clear_memory()

        # Test with increasing cell counts
        cell_counts = [100, 200, 400]
        memory_readings = []

        for max_cells in cell_counts:
            clear_memory()
            torch.cuda.reset_peak_memory_stats(device)

            B = 1
            n_cell_types = 4

            cells = torch.randn(
                B, n_cell_types, max_cells,
                small_model_config['n_genes'], device=device
            )
            cell_mask = torch.ones(
                B, n_cell_types, max_cells,
                dtype=torch.bool, device=device
            )

            with torch.no_grad():
                output, _, _ = transformer(cells, cell_mask)

            torch.cuda.synchronize()
            peak_memory = torch.cuda.max_memory_allocated(device)
            memory_readings.append(peak_memory / (1024 * 1024))  # MB

            del cells, cell_mask, output

        # Memory should increase but not explode
        # ISAB provides O(n*k) complexity where k is inducing points
        # So memory should scale sub-quadratically
        ratio_1_to_2 = memory_readings[1] / max(memory_readings[0], 0.1)
        ratio_2_to_4 = memory_readings[2] / max(memory_readings[1], 0.1)

        # The ratio should be less than 4 (linear would be 2, quadratic would be 4)
        assert ratio_2_to_4 < 4, (
            f"Memory scaling too aggressive: "
            f"100 cells={memory_readings[0]:.1f}MB, "
            f"400 cells={memory_readings[2]:.1f}MB"
        )

        clear_memory()

    def test_no_memory_leak_over_iterations(self, small_model_config, cpu_device):
        """No memory leak pattern over multiple forward passes."""
        from src.models.branches.pseudobulk_encoder import PseudobulkEncoder
        import gc

        encoder = PseudobulkEncoder(
            n_cell_types=small_model_config['n_cell_types'],
            n_genes=small_model_config['n_genes'],
            d_embed=small_model_config['d_embed'],
            dropout=0.0,
        ).to(cpu_device)
        encoder.eval()

        # Warmup
        x = torch.randn(4, small_model_config['n_cell_types'], small_model_config['n_genes'])
        with torch.no_grad():
            _ = encoder(x)
        del x
        gc.collect()

        # Track objects
        initial_objects = len(gc.get_objects())

        # Run many iterations
        for _ in range(20):
            x = torch.randn(4, small_model_config['n_cell_types'], small_model_config['n_genes'])
            with torch.no_grad():
                output = encoder(x)
            del x, output

        gc.collect()
        final_objects = len(gc.get_objects())

        # Object count should not grow significantly (allow some variance)
        growth = final_objects - initial_objects
        assert growth < 1000, f"Potential memory leak: object count grew by {growth}"


# ============================================================================
# Component-Level Scale Tests
# ============================================================================


class TestComponentScale:
    """Test individual components at scale."""

    def test_fusion_layer_scale(self, cpu_device):
        """FusionLayer handles large inputs."""
        from src.models.fusion.fusion_layer import FusionLayer

        B = 16
        n_cell_types = N_CELL_TYPES
        d_embed = 128
        d_fused = 128

        layer = FusionLayer(
            d_embed=d_embed,
            d_fused=d_fused,
            n_cell_types=n_cell_types,
        ).to(cpu_device)
        layer.eval()

        pseudobulk = torch.randn(B, n_cell_types, d_embed, device=cpu_device)
        hgt = torch.randn(B, n_cell_types, d_embed, device=cpu_device)
        cell = torch.randn(B, n_cell_types, d_embed, device=cpu_device)

        with torch.no_grad():
            output = layer(pseudobulk, hgt, cell)

        assert output.shape == (B, n_cell_types, d_fused)
        assert not torch.isnan(output).any()

    def test_pathology_attention_scale(self, cpu_device):
        """PathologyStratifiedAttention handles large inputs."""
        from src.models.fusion.pathology_attention import PathologyStratifiedAttention

        B = 16
        n_cell_types = N_CELL_TYPES
        d_fused = 128
        d_cond = 64
        n_heads = 8

        attention = PathologyStratifiedAttention(
            d_fused=d_fused,
            d_cond=d_cond,
            n_heads=n_heads,
            n_cell_types=n_cell_types,
        ).to(cpu_device)
        attention.eval()

        cell_type_embeddings = torch.randn(B, n_cell_types, d_fused, device=cpu_device)
        path_emb = torch.randn(B, d_cond, device=cpu_device)

        with torch.no_grad():
            attended, weights = attention(cell_type_embeddings, path_emb)

        assert attended.shape == (B, d_fused)
        assert weights.shape == (B, n_heads, n_cell_types)
        assert not torch.isnan(attended).any()
        assert not torch.isnan(weights).any()

    def test_set_transformer_scale(self, cpu_device):
        """SetTransformerEncoder handles large set sizes."""
        from src.models.components.set_transformer import SetTransformerEncoder

        B = 8
        set_size = 200
        d_input = 100
        d_model = 64

        encoder = SetTransformerEncoder(
            d_input=d_input,
            d_model=d_model,
            n_heads=4,
            n_isab_layers=2,
            n_inducing=16,
            dropout=0.0,
        ).to(cpu_device)
        encoder.eval()

        x = torch.randn(B, set_size, d_input, device=cpu_device)
        mask = torch.ones(B, set_size, dtype=torch.bool, device=cpu_device)

        with torch.no_grad():
            output, attention = encoder(x, mask)

        assert output.shape == (B, d_model)
        assert not torch.isnan(output).any()

    def test_region_handler_scale(self, cpu_device):
        """RegionHandler handles large inputs."""
        from src.models.components.region_handler import RegionHandler

        B = 16
        n_regions = N_REGIONS
        n_cell_types = N_CELL_TYPES
        d_model = 128

        handler = RegionHandler(d_model=d_model, n_regions=n_regions).to(cpu_device)
        handler.eval()

        x = torch.randn(B, n_regions, n_cell_types, d_model, device=cpu_device)
        region_mask = torch.ones(B, n_regions, dtype=torch.bool, device=cpu_device)

        with torch.no_grad():
            pooled, region_context = handler(x, region_mask)

        assert pooled.shape == (B, n_cell_types, d_model)
        assert region_context.shape == (B, d_model)
        assert not torch.isnan(pooled).any()
        assert not torch.isnan(region_context).any()


# ============================================================================
# GPU-Specific Scale Tests
# ============================================================================


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="GPU tests require CUDA"
)
class TestGPUScale:
    """GPU-specific scale tests."""

    def test_large_batch_gpu(self, small_model_config):
        """Large batch on GPU works correctly."""
        from src.models.full_model import CognitiveResilienceModel

        device = torch.device("cuda:0")

        if not check_memory_available(3000):
            pytest.skip("Insufficient GPU memory")

        model = CognitiveResilienceModel(**small_model_config, use_bayesian_head=False)
        model = model.to(device)
        model.eval()

        batch_size = 32
        inputs = create_sample_inputs(
            batch_size=batch_size,
            n_genes=small_model_config['n_genes'],
            device=device,
        )

        with torch.no_grad():
            output = model(**inputs)

        assert output['mean'].shape == (batch_size, 1)
        assert not torch.isnan(output['mean']).any()

        clear_memory()

    def test_training_step_scale(self, small_model_config):
        """Training step at scale completes without issues."""
        from src.models.full_model import CognitiveResilienceModel

        device = torch.device("cuda:0")

        if not check_memory_available(4000):
            pytest.skip("Insufficient GPU memory")

        model = CognitiveResilienceModel(**small_model_config, use_bayesian_head=False)
        model = model.to(device)
        model.train()

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        batch_size = 16
        inputs = create_sample_inputs(
            batch_size=batch_size,
            n_genes=small_model_config['n_genes'],
            device=device,
        )
        target = torch.randn(batch_size, 1, device=device)

        # Forward pass
        output = model(**inputs)

        # Compute loss
        loss = nn.functional.mse_loss(output['mean'], target)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Check gradients
        for name, param in model.named_parameters():
            if param.grad is not None:
                assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"

        # Optimizer step
        optimizer.step()

        # Check parameters after update
        for name, param in model.named_parameters():
            assert not torch.isnan(param).any(), f"NaN parameter after update: {name}"

        clear_memory()

    def test_mixed_precision_scale(self, small_model_config):
        """Mixed precision training at scale works correctly."""
        from src.models.full_model import CognitiveResilienceModel

        device = torch.device("cuda:0")

        if not check_memory_available(2000):
            pytest.skip("Insufficient GPU memory")

        model = CognitiveResilienceModel(**small_model_config, use_bayesian_head=False)
        model = model.to(device)
        model.train()

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scaler = torch.amp.GradScaler('cuda')

        batch_size = 16
        inputs = create_sample_inputs(
            batch_size=batch_size,
            n_genes=small_model_config['n_genes'],
            device=device,
        )
        target = torch.randn(batch_size, 1, device=device)

        # Forward pass with autocast
        with torch.amp.autocast('cuda'):
            output = model(**inputs)
            loss = nn.functional.mse_loss(output['mean'], target)

        # Backward pass with scaler
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Verify no NaN in output or parameters
        assert not torch.isnan(output['mean']).any()

        for name, param in model.named_parameters():
            assert not torch.isnan(param).any(), f"NaN parameter after update: {name}"

        clear_memory()


# ============================================================================
# Stress Tests for Edge Cases at Scale
# ============================================================================


class TestScaleEdgeCases:
    """Edge cases that might appear at scale."""

    def test_all_cells_masked(self, small_model_config, cpu_device):
        """Model handles case where all cells are masked in some cell types."""
        from src.models.branches.cell_transformer import CellTransformer

        transformer = CellTransformer(
            n_genes=small_model_config['n_genes'],
            n_cell_types=small_model_config['n_cell_types'],
            d_model=small_model_config['d_embed'],
            n_heads=small_model_config['n_hgt_heads'],
            n_isab_layers=small_model_config['n_isab_layers'],
            n_inducing=small_model_config['n_inducing_points'],
            dropout=0.0,
        ).to(cpu_device)
        transformer.eval()

        B = 2
        max_cells = 50
        cells = torch.randn(
            B, small_model_config['n_cell_types'], max_cells,
            small_model_config['n_genes'], device=cpu_device
        )

        # Mask out half of the cell types completely
        cell_mask = torch.ones(
            B, small_model_config['n_cell_types'], max_cells,
            dtype=torch.bool, device=cpu_device
        )
        cell_mask[:, ::2, :] = False  # Every other cell type fully masked

        with torch.no_grad():
            output, selection_weights, _ = transformer(cells, cell_mask)

        # Should still produce valid output
        assert not torch.isnan(output).any()

    def test_all_regions_masked_except_one(self, small_model_config, cpu_device):
        """Model handles case where only one region is available."""
        from src.models.components.region_handler import RegionHandler

        handler = RegionHandler(
            d_model=small_model_config['d_embed'],
            n_regions=small_model_config['n_regions'],
        ).to(cpu_device)
        handler.eval()

        B = 4
        n_regions = small_model_config['n_regions']
        n_cell_types = small_model_config['n_cell_types']
        d_model = small_model_config['d_embed']

        x = torch.randn(B, n_regions, n_cell_types, d_model, device=cpu_device)

        # Only first region available
        region_mask = torch.zeros(B, n_regions, dtype=torch.bool, device=cpu_device)
        region_mask[:, 0] = True

        with torch.no_grad():
            pooled, region_context = handler(x, region_mask)

        assert not torch.isnan(pooled).any()
        assert not torch.isnan(region_context).any()

    def test_highly_sparse_attention(self, cpu_device):
        """Attention works with very sparse masks."""
        from src.models.components.set_transformer import SetTransformerEncoder

        encoder = SetTransformerEncoder(
            d_input=64,
            d_model=32,
            n_heads=4,
            n_isab_layers=1,
            n_inducing=8,
            dropout=0.0,
        ).to(cpu_device)
        encoder.eval()

        B = 4
        set_size = 100

        x = torch.randn(B, set_size, 64, device=cpu_device)

        # Very sparse mask - only 5 elements valid
        mask = torch.zeros(B, set_size, dtype=torch.bool, device=cpu_device)
        mask[:, :5] = True

        with torch.no_grad():
            output, attention = encoder(x, mask)

        assert not torch.isnan(output).any()

    def test_varying_sequence_lengths_in_batch(self, cpu_device):
        """Model handles varying sequence lengths within a batch."""
        from src.models.components.set_transformer import SetTransformerEncoder

        encoder = SetTransformerEncoder(
            d_input=64,
            d_model=32,
            n_heads=4,
            n_isab_layers=1,
            n_inducing=8,
            dropout=0.0,
        ).to(cpu_device)
        encoder.eval()

        B = 4
        max_set_size = 100

        x = torch.randn(B, max_set_size, 64, device=cpu_device)

        # Different lengths per sample
        mask = torch.zeros(B, max_set_size, dtype=torch.bool, device=cpu_device)
        lengths = [10, 50, 30, 80]
        for i, length in enumerate(lengths):
            mask[i, :length] = True

        with torch.no_grad():
            output, attention = encoder(x, mask)

        assert output.shape == (B, 32)
        assert not torch.isnan(output).any()
