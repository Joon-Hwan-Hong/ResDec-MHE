"""
GPU/CUDA tests for all model components.

Tests verify that all model components work correctly on GPU, with proper skipif
decorators for when CUDA is unavailable. The tests cover:

1. GPU Device Tests - Model to CUDA, parameters on CUDA, forward/backward pass
2. Per-Component GPU Tests - Individual component verification on GPU
3. Multi-GPU Tests - DataParallel and multi-device scenarios
4. Memory Tests - Memory management and leak detection

Usage:
    # Run all CUDA tests
    pytest tests/unit/models/test_gpu_cuda.py -v

    # Run only with cuda marker
    pytest -m cuda tests/unit/models/test_gpu_cuda.py -v

    # Skip multi-GPU tests (for single GPU machines)
    pytest tests/unit/models/test_gpu_cuda.py -v -k "not multi_gpu"
"""

import gc

import pytest
import torch
import torch.nn as nn

# Skip entire module if CUDA is not available
pytestmark = pytest.mark.cuda


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def cuda_device():
    """Provide CUDA device if available."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda:0")


@pytest.fixture
def second_cuda_device():
    """Provide second CUDA device if available."""
    if torch.cuda.device_count() < 2:
        pytest.skip("Need 2+ GPUs for this test")
    return torch.device("cuda:1")


@pytest.fixture
def small_model_config():
    """Small model configuration for fast GPU testing."""
    return {
        'n_genes': 50,
        'n_cell_types': 31,
        'd_embed': 32,
        'd_fused': 32,
        'd_cond': 16,
        'n_regions': 6,
        'n_hgt_layers': 1,
        'n_hgt_heads': 4,
        'n_isab_layers': 1,
        'n_inducing_points': 4,
        'n_attention_heads': 4,
        'd_head_hidden': 16,
        'dropout': 0.0,
    }


@pytest.fixture
def sample_inputs(cuda_device):
    """Create sample inputs on CUDA device."""
    from src.data.constants import CELL_TYPE_ORDER, ALL_EDGE_TYPES, sanitize_key

    B = 2
    n_regions = 6
    n_cell_types = 31
    n_genes = 50
    max_cells = 10

    edge_index_dict_list = []
    edge_attr_dict_list = []
    for _ in range(B):
        edge_index_dict = {}
        edge_attr_dict = {}
        for src_ct in CELL_TYPE_ORDER[:3]:
            for dst_ct in CELL_TYPE_ORDER[:3]:
                for et in ALL_EDGE_TYPES[:2]:
                    key = (sanitize_key(src_ct), sanitize_key(et), sanitize_key(dst_ct))
                    edge_index_dict[key] = torch.zeros(2, 5, dtype=torch.long, device=cuda_device)
                    edge_attr_dict[key] = torch.rand(5, 1, device=cuda_device)
        edge_index_dict_list.append(edge_index_dict)
        edge_attr_dict_list.append(edge_attr_dict)

    return {
        'region_pseudobulk': torch.randn(B, n_regions, n_cell_types, n_genes, device=cuda_device),
        'region_mask': torch.ones(B, n_regions, dtype=torch.bool, device=cuda_device),
        'edge_index_dict_list': edge_index_dict_list,
        'edge_attr_dict_list': edge_attr_dict_list,
        'cells': torch.randn(B, n_cell_types, max_cells, n_genes, device=cuda_device),
        'cell_mask': torch.ones(B, n_cell_types, max_cells, dtype=torch.bool, device=cuda_device),
        'pathology': torch.randn(B, 3, device=cuda_device),
        'cognition': torch.randn(B, 1, device=cuda_device),
    }


def move_inputs_to_device(inputs: dict, device: torch.device) -> dict:
    """Helper to move all input tensors to a device."""
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}


def clear_cuda_memory():
    """Helper to clear CUDA memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.synchronize()


# ─────────────────────────────────────────────────────────────────────────────
# GPU Device Tests
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestModelToGPU:
    """Test moving CognitiveResilienceModel to GPU."""

    def test_model_to_cuda(self, small_model_config, cuda_device):
        """Verify CognitiveResilienceModel can be moved to CUDA."""
        from src.models.full_model import CognitiveResilienceModel

        model = CognitiveResilienceModel(**small_model_config, use_bayesian_head=False)
        model = model.to(cuda_device)

        # Verify model is on CUDA
        assert next(model.parameters()).device.type == "cuda"

    def test_model_parameters_on_cuda(self, small_model_config, cuda_device):
        """Verify all parameters are on CUDA after .cuda()."""
        from src.models.full_model import CognitiveResilienceModel

        model = CognitiveResilienceModel(**small_model_config, use_bayesian_head=False)
        model = model.cuda()

        # Check ALL parameters are on CUDA
        for name, param in model.named_parameters():
            assert param.device.type == "cuda", f"Parameter {name} not on CUDA"

    def test_forward_pass_on_cuda(self, small_model_config, cuda_device, sample_inputs):
        """Verify forward pass works on CUDA."""
        from src.models.full_model import CognitiveResilienceModel

        model = CognitiveResilienceModel(**small_model_config, use_bayesian_head=False)
        model = model.to(cuda_device)
        model.eval()

        with torch.no_grad():
            output = model(**sample_inputs)

        # Verify outputs are on CUDA
        assert output['mean'].device.type == "cuda"
        assert output['attention_weights'].device.type == "cuda"

        # Verify output shapes
        assert output['mean'].shape == (2, 1)

    def test_backward_pass_on_cuda(self, small_model_config, cuda_device, sample_inputs):
        """Verify gradients flow on CUDA for used parameters."""
        from src.models.full_model import CognitiveResilienceModel

        model = CognitiveResilienceModel(**small_model_config, use_bayesian_head=False)
        model = model.to(cuda_device)
        model.train()

        output = model(**sample_inputs)
        loss = output['mean'].sum()
        loss.backward()

        # Count parameters with gradients
        params_with_grad = 0
        params_without_grad = 0
        for name, param in model.named_parameters():
            if param.requires_grad:
                if param.grad is not None:
                    params_with_grad += 1
                    assert param.grad.device.type == "cuda", f"Gradient for {name} not on CUDA"
                else:
                    params_without_grad += 1

        # Most parameters should have gradients
        # Note: HGT has many per-node-type and per-edge-type parameters that won't
        # receive gradients when those types aren't involved in any edges.
        # With 31 node types and 5 edge types, most HGT type-specific params get no grads.
        assert params_with_grad > 0, "No parameters received gradients"
        # At least 30% of parameters should have gradients (lower due to HGT type params)
        total = params_with_grad + params_without_grad
        assert params_with_grad / total > 0.3, (
            f"Too few parameters with gradients: {params_with_grad}/{total}"
        )

    def test_cuda_determinism(self, small_model_config, cuda_device, sample_inputs):
        """Verify deterministic output on CUDA with torch.use_deterministic_algorithms."""
        from src.models.full_model import CognitiveResilienceModel

        # Set deterministic mode
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        model = CognitiveResilienceModel(**small_model_config, use_bayesian_head=False)
        model = model.to(cuda_device)
        model.eval()

        # Run twice with same seed
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        with torch.no_grad():
            output1 = model(**sample_inputs)

        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        with torch.no_grad():
            output2 = model(**sample_inputs)

        # Results should be identical
        assert torch.allclose(output1['mean'], output2['mean'], atol=1e-6)
        assert torch.allclose(output1['attention_weights'], output2['attention_weights'], atol=1e-6)

        # Reset to default
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


# ─────────────────────────────────────────────────────────────────────────────
# Per-Component GPU Tests
# ─────────────────────────────────────────────────────────────────────────────


# ── Factory functions for parametrized component CUDA tests ──────────────────

N_CT = 31   # local shorthand matching small_model_config
N_REG = 6


def _make_fusion_layer():
    from src.models.fusion.fusion_layer import FusionLayer
    return FusionLayer(d_embed=64, d_fused=128, n_cell_types=N_CT)


def _fusion_inputs(device):
    return (
        torch.randn(4, N_CT, 64, device=device, requires_grad=True),
        torch.randn(4, N_CT, 64, device=device, requires_grad=True),
        torch.randn(4, N_CT, 64, device=device, requires_grad=True),
    )


def _make_pathology_encoder():
    from src.models.fusion.pathology_encoder import PathologyEncoder
    return PathologyEncoder(n_pathology_features=3, d_region=128, d_cond=64)


def _pathology_encoder_inputs(device):
    return (
        torch.randn(4, 3, device=device, requires_grad=True),
        torch.randn(4, 128, device=device, requires_grad=True),
    )


def _make_pathology_attention():
    from src.models.fusion.pathology_attention import PathologyStratifiedAttention
    return PathologyStratifiedAttention(
        d_fused=64, d_cond=32, n_heads=4, n_cell_types=N_CT,
    )


def _pathology_attention_inputs(device):
    return (
        torch.randn(4, N_CT, 64, device=device, requires_grad=True),
        torch.randn(4, 32, device=device, requires_grad=True),
    )


def _make_region_handler():
    from src.models.components.region_handler import RegionHandler
    return RegionHandler(d_model=128, n_regions=N_REG)


def _region_handler_inputs(device):
    return (
        torch.randn(4, N_REG, N_CT, 128, device=device, requires_grad=True),
        torch.ones(4, N_REG, dtype=torch.bool, device=device),
    )


def _make_deterministic_head():
    from src.models.heads.deterministic_head import DeterministicPredictionHead
    return DeterministicPredictionHead(d_input=128, d_hidden=64)


def _deterministic_head_inputs(device):
    return (
        torch.randn(4, 128, device=device, requires_grad=True),
    )


COMPONENT_CUDA_CASES = [
    ("FusionLayer", _make_fusion_layer, _fusion_inputs),
    ("PathologyEncoder", _make_pathology_encoder, _pathology_encoder_inputs),
    ("PathologyAttention", _make_pathology_attention, _pathology_attention_inputs),
    ("RegionHandler", _make_region_handler, _region_handler_inputs),
    ("DeterministicHead", _make_deterministic_head, _deterministic_head_inputs),
]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestComponentCUDAParametrized:
    """Parametrized CUDA tests for simple components.

    Each case creates a component, moves it to GPU, runs a forward pass,
    and verifies output is on CUDA, finite, and that gradients flow back.
    """

    @pytest.mark.parametrize(
        "name,make_component,make_inputs",
        COMPONENT_CUDA_CASES,
        ids=[c[0] for c in COMPONENT_CUDA_CASES],
    )
    def test_forward_on_cuda(self, cuda_device, name, make_component, make_inputs):
        """Component forward and backward on GPU."""
        component = make_component().to(cuda_device)
        inputs = make_inputs(cuda_device)

        output = component(*inputs)

        # Unpack output: may be a single tensor or a tuple
        if isinstance(output, tuple):
            out_tensor = output[0]
        elif isinstance(output, dict):
            out_tensor = next(iter(output.values()))
        else:
            out_tensor = output

        # Verify output is on CUDA and finite
        assert out_tensor.device.type == "cuda", f"{name} output not on CUDA"
        assert torch.isfinite(out_tensor).all(), f"{name} output contains non-finite values"

        # Verify gradients flow to at least one grad-requiring input
        if isinstance(output, tuple):
            loss = sum(o.sum() for o in output if isinstance(o, torch.Tensor))
        else:
            loss = out_tensor.sum()
        loss.backward()

        grad_inputs = [inp for inp in inputs if isinstance(inp, torch.Tensor) and inp.requires_grad]
        assert any(inp.grad is not None for inp in grad_inputs), (
            f"{name}: no gradients flowed to any input"
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestBayesianHeadCUDA:
    """Test BayesianPredictionHead on GPU.

    Note: Pyro's PyroSample has a known limitation where prior distributions
    are created at __init__ time on CPU and don't automatically move to CUDA
    when the module is moved. Full GPU support for BayesianPredictionHead
    requires either:
    1. Using a device-aware custom guide
    2. Modifying BayesianPredictionHead to use device-aware priors
    3. Using Pyro's to() override (requires Pyro 1.9+)

    The deterministic portions of the head (fc_log_std) work correctly on GPU.
    """

    def test_bayesian_head_deterministic_portion_cuda(self, cuda_device):
        """Test that deterministic parts of BayesianPredictionHead work on GPU.

        The fc_log_std layer (which learns aleatoric uncertainty) is a regular
        nn.Linear and should work correctly on GPU.
        """
        from src.models.heads.bayesian_head import BayesianPredictionHead

        head = BayesianPredictionHead(d_input=128, d_hidden=64).to(cuda_device)

        # Verify fc_log_std is on GPU
        assert head.fc_log_std.weight.device.type == "cuda"
        assert head.fc_log_std.bias.device.type == "cuda"

        # Verify we can compute through fc_log_std
        x = torch.randn(4, 64, device=cuda_device)  # After fc2 output
        log_std_out = head.fc_log_std(x)
        assert log_std_out.device.type == "cuda"
        assert log_std_out.shape == (4, 1)

    def test_bayesian_head_parameter_count(self, cuda_device):
        """Test Bayesian head has expected parameter count on GPU."""
        from src.models.heads.bayesian_head import BayesianPredictionHead

        head = BayesianPredictionHead(d_input=128, d_hidden=64).to(cuda_device)

        # Count parameters (note: PyroSample wraps parameters)
        # The fc_log_std is a regular parameter, others are PyroSample
        param_count = sum(p.numel() for p in head.parameters())

        # Expected: fc_log_std has 64*1 + 1 = 65 parameters
        # The PyroSample parameters are handled differently
        assert param_count >= 65, "Expected at least fc_log_std parameters"


# ─────────────────────────────────────────────────────────────────────────────
# Multi-GPU Tests
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.skipif(
    torch.cuda.device_count() < 2,
    reason="Need 2+ GPUs for multi-GPU tests"
)
class TestMultiGPU:
    """Test multi-GPU functionality."""

    def test_model_on_different_gpus(self, small_model_config):
        """Model works on GPU:0 and GPU:1."""
        from src.models.full_model import CognitiveResilienceModel

        B = 2
        n_genes = 50
        n_cell_types = 31

        def create_inputs(device):
            from src.data.constants import CELL_TYPE_ORDER, ALL_EDGE_TYPES, sanitize_key

            edge_index_dict_list = []
            edge_attr_dict_list = []
            for _ in range(B):
                eid = {}
                ead = {}
                for src_ct in CELL_TYPE_ORDER[:3]:
                    for dst_ct in CELL_TYPE_ORDER[:3]:
                        for et in ALL_EDGE_TYPES[:2]:
                            key = (sanitize_key(src_ct), sanitize_key(et), sanitize_key(dst_ct))
                            eid[key] = torch.zeros(2, 5, dtype=torch.long, device=device)
                            ead[key] = torch.rand(5, 1, device=device)
                edge_index_dict_list.append(eid)
                edge_attr_dict_list.append(ead)

            return {
                'region_pseudobulk': torch.randn(B, 6, n_cell_types, n_genes, device=device),
                'region_mask': torch.ones(B, 6, dtype=torch.bool, device=device),
                'edge_index_dict_list': edge_index_dict_list,
                'edge_attr_dict_list': edge_attr_dict_list,
                'cells': torch.randn(B, n_cell_types, 10, n_genes, device=device),
                'cell_mask': torch.ones(B, n_cell_types, 10, dtype=torch.bool, device=device),
                'pathology': torch.randn(B, 3, device=device),
            }

        # Test on GPU:0
        device0 = torch.device("cuda:0")
        model0 = CognitiveResilienceModel(**small_model_config, use_bayesian_head=False)
        model0 = model0.to(device0)
        inputs0 = create_inputs(device0)

        with torch.no_grad():
            output0 = model0(**inputs0)

        assert output0['mean'].device == device0

        # Test on GPU:1
        device1 = torch.device("cuda:1")
        model1 = CognitiveResilienceModel(**small_model_config, use_bayesian_head=False)
        model1 = model1.to(device1)
        inputs1 = create_inputs(device1)

        with torch.no_grad():
            output1 = model1(**inputs1)

        assert output1['mean'].device == device1

    def test_component_data_parallel(self):
        """nn.DataParallel works with individual components.

        Note: The full CognitiveResilienceModel has complex edge handling
        that doesn't work well with DataParallel's tensor splitting.
        Testing components individually instead.
        """
        from src.models.fusion.fusion_layer import FusionLayer

        layer = FusionLayer(d_embed=64, d_fused=128, n_cell_types=31).cuda()
        parallel_layer = nn.DataParallel(layer, device_ids=[0, 1])

        # Create inputs with larger batch size for parallelism
        B = 8  # Larger batch to be split across GPUs
        pseudobulk = torch.randn(B, 31, 64).cuda()
        hgt = torch.randn(B, 31, 64).cuda()
        cell = torch.randn(B, 31, 64).cuda()

        with torch.no_grad():
            output = parallel_layer(pseudobulk, hgt, cell)

        # Output should be on cuda:0 (primary device)
        assert output.device.index == 0
        assert output.shape == (B, 31, 128)


# ─────────────────────────────────────────────────────────────────────────────
# Memory Tests
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestCUDAMemory:
    """Test CUDA memory management."""

    def test_cuda_memory_cleared_after_forward(self, small_model_config, cuda_device):
        """Memory is properly freed after forward pass."""
        from src.models.full_model import CognitiveResilienceModel

        clear_cuda_memory()
        initial_memory = torch.cuda.memory_allocated(cuda_device)

        model = CognitiveResilienceModel(**small_model_config, use_bayesian_head=False)
        model = model.to(cuda_device)

        B = 2
        n_genes = 50
        n_cell_types = 31

        from src.data.constants import CELL_TYPE_ORDER, ALL_EDGE_TYPES, sanitize_key

        edge_index_dict_list = []
        edge_attr_dict_list = []
        for _ in range(B):
            eid = {}
            ead = {}
            for src_ct in CELL_TYPE_ORDER[:3]:
                for dst_ct in CELL_TYPE_ORDER[:3]:
                    for et in ALL_EDGE_TYPES[:2]:
                        key = (sanitize_key(src_ct), sanitize_key(et), sanitize_key(dst_ct))
                        eid[key] = torch.zeros(2, 5, dtype=torch.long, device=cuda_device)
                        ead[key] = torch.rand(5, 1, device=cuda_device)
            edge_index_dict_list.append(eid)
            edge_attr_dict_list.append(ead)

        inputs = {
            'region_pseudobulk': torch.randn(B, 6, n_cell_types, n_genes, device=cuda_device),
            'region_mask': torch.ones(B, 6, dtype=torch.bool, device=cuda_device),
            'edge_index_dict_list': edge_index_dict_list,
            'edge_attr_dict_list': edge_attr_dict_list,
            'cells': torch.randn(B, n_cell_types, 10, n_genes, device=cuda_device),
            'cell_mask': torch.ones(B, n_cell_types, 10, dtype=torch.bool, device=cuda_device),
            'pathology': torch.randn(B, 3, device=cuda_device),
        }

        # Forward pass
        with torch.no_grad():
            output = model(**inputs)

        # Delete everything
        del output, inputs, model
        clear_cuda_memory()

        final_memory = torch.cuda.memory_allocated(cuda_device)

        # Memory should be back to approximately initial level
        # Allow some tolerance for CUDA context overhead
        memory_diff = final_memory - initial_memory
        assert memory_diff < 10 * 1024 * 1024, f"Memory leak detected: {memory_diff / 1024 / 1024:.2f} MB"

    def test_no_memory_leak_over_multiple_batches(self, small_model_config, cuda_device):
        """No memory leak pattern over multiple forward passes."""
        from src.models.full_model import CognitiveResilienceModel

        clear_cuda_memory()

        model = CognitiveResilienceModel(**small_model_config, use_bayesian_head=False)
        model = model.to(cuda_device)
        model.eval()

        B = 2
        n_genes = 50
        n_cell_types = 31

        from src.data.constants import CELL_TYPE_ORDER, ALL_EDGE_TYPES, sanitize_key

        def create_batch():
            edge_index_dict_list = []
            edge_attr_dict_list = []
            for _ in range(B):
                eid = {}
                ead = {}
                for src_ct in CELL_TYPE_ORDER[:3]:
                    for dst_ct in CELL_TYPE_ORDER[:3]:
                        for et in ALL_EDGE_TYPES[:2]:
                            key = (sanitize_key(src_ct), sanitize_key(et), sanitize_key(dst_ct))
                            eid[key] = torch.zeros(2, 5, dtype=torch.long, device=cuda_device)
                            ead[key] = torch.rand(5, 1, device=cuda_device)
                edge_index_dict_list.append(eid)
                edge_attr_dict_list.append(ead)

            return {
                'region_pseudobulk': torch.randn(B, 6, n_cell_types, n_genes, device=cuda_device),
                'region_mask': torch.ones(B, 6, dtype=torch.bool, device=cuda_device),
                'edge_index_dict_list': edge_index_dict_list,
                'edge_attr_dict_list': edge_attr_dict_list,
                'cells': torch.randn(B, n_cell_types, 10, n_genes, device=cuda_device),
                'cell_mask': torch.ones(B, n_cell_types, 10, dtype=torch.bool, device=cuda_device),
                'pathology': torch.randn(B, 3, device=cuda_device),
            }

        # Warmup run
        with torch.no_grad():
            inputs = create_batch()
            _ = model(**inputs)
            del inputs
        clear_cuda_memory()

        # Record memory after warmup
        baseline_memory = torch.cuda.memory_allocated(cuda_device)
        memory_readings = []

        # Run multiple batches
        n_batches = 10
        for _ in range(n_batches):
            with torch.no_grad():
                inputs = create_batch()
                output = model(**inputs)
                del output, inputs

            clear_cuda_memory()
            memory_readings.append(torch.cuda.memory_allocated(cuda_device))

        # Check for memory growth
        # Memory should stay relatively constant
        max_memory = max(memory_readings)
        min_memory = min(memory_readings)
        memory_variance = max_memory - min_memory

        # Allow up to 5MB variance (for CUDA allocator fluctuations)
        assert memory_variance < 5 * 1024 * 1024, (
            f"Memory fluctuation too large: {memory_variance / 1024 / 1024:.2f} MB"
        )

        # Check no consistent growth (leak)
        # Average of first half vs second half should be similar
        first_half_avg = sum(memory_readings[:5]) / 5
        second_half_avg = sum(memory_readings[5:]) / 5
        growth = second_half_avg - first_half_avg

        # Allow up to 1MB growth
        assert growth < 1 * 1024 * 1024, (
            f"Memory growing over time: {growth / 1024 / 1024:.2f} MB"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Additional Component GPU Tests
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestBranchEncodersCUDA:
    """Test branch encoders on GPU."""

    def test_pseudobulk_encoder_cuda(self, cuda_device):
        """PseudobulkEncoder forward and backward on GPU."""
        from src.models.branches.pseudobulk_encoder import PseudobulkEncoder

        encoder = PseudobulkEncoder(
            n_cell_types=31,
            n_genes=50,
            d_embed=64,
            dropout=0.0,
        ).to(cuda_device)

        x = torch.randn(4, 31, 50, device=cuda_device, requires_grad=True)

        output = encoder(x)

        # Verify output on CUDA
        assert output.device.type == "cuda"
        assert output.shape == (4, 31, 64)

        # Verify gradients flow
        loss = output.sum()
        loss.backward()

        assert x.grad is not None

    def test_cell_transformer_cuda(self, cuda_device):
        """CellTransformer forward and backward on GPU."""
        from src.models.branches.cell_transformer import CellTransformer

        transformer = CellTransformer(
            n_genes=50,
            n_cell_types=31,
            d_model=64,
            n_heads=4,
            n_isab_layers=1,
            n_inducing=8,
            dropout=0.0,
        ).to(cuda_device)

        cells = torch.randn(4, 31, 10, 50, device=cuda_device, requires_grad=True)
        cell_mask = torch.ones(4, 31, 10, dtype=torch.bool, device=cuda_device)

        output, selection_weights, _ = transformer(cells, cell_mask)

        # Verify outputs on CUDA
        assert output.device.type == "cuda"
        assert output.shape == (4, 31, 64)

        # Verify gradients flow
        loss = output.sum()
        loss.backward()

        assert cells.grad is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestSetTransformerCUDA:
    """Test SetTransformer components on GPU."""

    def test_set_transformer_encoder_cuda(self, cuda_device):
        """SetTransformerEncoder forward and backward on GPU."""
        from src.models.components.set_transformer import SetTransformerEncoder

        encoder = SetTransformerEncoder(
            d_input=50,
            d_model=64,
            n_heads=4,
            n_isab_layers=1,
            n_inducing=8,
            dropout=0.0,
        ).to(cuda_device)

        x = torch.randn(4, 20, 50, device=cuda_device, requires_grad=True)
        mask = torch.ones(4, 20, dtype=torch.bool, device=cuda_device)

        output, attention = encoder(x, mask)

        # Verify output on CUDA
        assert output.device.type == "cuda"
        assert output.shape == (4, 64)

        # Verify gradients flow
        loss = output.sum()
        loss.backward()

        assert x.grad is not None

    def test_isab_cuda(self, cuda_device):
        """ISAB forward and backward on GPU."""
        from src.models.components.set_transformer import ISAB

        isab = ISAB(
            d_model=64,
            n_heads=4,
            n_inducing=8,
            dropout=0.0,
        ).to(cuda_device)

        x = torch.randn(4, 20, 64, device=cuda_device, requires_grad=True)
        mask = torch.ones(4, 20, dtype=torch.bool, device=cuda_device)

        output = isab(x, mask)

        # Verify output on CUDA
        assert output.device.type == "cuda"
        assert output.shape == (4, 20, 64)

        # Verify gradients flow
        loss = output.sum()
        loss.backward()

        assert x.grad is not None

    def test_pma_cuda(self, cuda_device):
        """PMA forward and backward on GPU."""
        from src.models.components.set_transformer import PMA

        pma = PMA(
            d_model=64,
            n_heads=4,
            n_seeds=1,
            dropout=0.0,
        ).to(cuda_device)

        x = torch.randn(4, 20, 64, device=cuda_device, requires_grad=True)
        mask = torch.ones(4, 20, dtype=torch.bool, device=cuda_device)

        output, attention = pma(x, mask)

        # Verify output on CUDA
        assert output.device.type == "cuda"
        assert output.shape == (4, 1, 64)

        # Verify gradients flow
        loss = output.sum()
        loss.backward()

        assert x.grad is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestCellTypeSelectorCUDA:
    """Test CellTypeSelector on GPU."""

    def test_cell_type_selector_cuda(self, cuda_device):
        """CellTypeSelector forward on GPU."""
        from src.models.components.cell_type_selector import CellTypeSelector

        selector = CellTypeSelector(
            n_cell_types=31,
            temperature=1.0,
        ).to(cuda_device)

        # Forward pass - get selection weights
        weights = selector.get_selection_weights()

        # Verify output on CUDA
        assert weights.device.type == "cuda"
        assert weights.shape == (31,)
        assert torch.allclose(weights.sum(), torch.tensor(1.0, device=cuda_device), atol=1e-5)

    def test_cell_type_selector_gradient_cuda(self, cuda_device):
        """CellTypeSelector gradients flow on GPU."""
        from src.models.components.cell_type_selector import CellTypeSelector

        selector = CellTypeSelector(
            n_cell_types=31,
            temperature=1.0,
        ).to(cuda_device)

        weights = selector.get_selection_weights()
        loss = weights.sum()
        loss.backward()

        assert selector.selection_logits.grad is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestGeneAttentionGateCUDA:
    """Test GeneAttentionGate on GPU."""

    def test_gene_attention_gate_cuda(self, cuda_device):
        """GeneAttentionGate forward and backward on GPU."""
        from src.models.components.gene_attention_gate import GeneAttentionGate

        gate = GeneAttentionGate(
            n_cell_types=31,
            n_genes=50,
            temperature=2.0,
        ).to(cuda_device)

        x = torch.randn(4, 31, 50, device=cuda_device, requires_grad=True)

        output = gate(x)

        # Verify outputs on CUDA
        assert output.device.type == "cuda"
        assert output.shape == (4, 31, 50)

        # Verify gradients flow
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert gate.gate_logits.grad is not None

    def test_gene_attention_gate_weights_cuda(self, cuda_device):
        """GeneAttentionGate weights are properly on GPU."""
        from src.models.components.gene_attention_gate import GeneAttentionGate

        gate = GeneAttentionGate(
            n_cell_types=31,
            n_genes=50,
            temperature=2.0,
        ).to(cuda_device)

        weights = gate.get_gate_weights()

        # Verify weights on CUDA
        assert weights.device.type == "cuda"
        assert weights.shape == (31, 50)

        # Weights should sum to 1 per cell type
        row_sums = weights.sum(dim=1)
        assert torch.allclose(row_sums, torch.ones(31, device=cuda_device), atol=1e-5)


# ─────────────────────────────────────────────────────────────────────────────
# CPU/GPU Consistency Tests
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestCPUGPUConsistency:
    """Test that CPU and GPU produce consistent results."""

    def test_fusion_layer_cpu_gpu_consistency(self, cuda_device):
        """FusionLayer produces same results on CPU and GPU."""
        from src.models.fusion.fusion_layer import FusionLayer

        torch.manual_seed(42)
        layer_cpu = FusionLayer(d_embed=64, d_fused=128, n_cell_types=31)

        torch.manual_seed(42)
        layer_gpu = FusionLayer(d_embed=64, d_fused=128, n_cell_types=31).to(cuda_device)

        # Same inputs
        torch.manual_seed(123)
        pb = torch.randn(4, 31, 64)
        hgt = torch.randn(4, 31, 64)
        cell = torch.randn(4, 31, 64)

        # CPU forward
        layer_cpu.eval()
        with torch.no_grad():
            output_cpu = layer_cpu(pb, hgt, cell)

        # GPU forward
        layer_gpu.eval()
        with torch.no_grad():
            output_gpu = layer_gpu(
                pb.to(cuda_device),
                hgt.to(cuda_device),
                cell.to(cuda_device)
            )

        # Compare (bring GPU output to CPU)
        assert torch.allclose(output_cpu, output_gpu.cpu(), atol=1e-5)

    def test_deterministic_head_cpu_gpu_consistency(self, cuda_device):
        """DeterministicPredictionHead produces same results on CPU and GPU."""
        from src.models.heads.deterministic_head import DeterministicPredictionHead

        torch.manual_seed(42)
        head_cpu = DeterministicPredictionHead(d_input=128, d_hidden=64)

        torch.manual_seed(42)
        head_gpu = DeterministicPredictionHead(d_input=128, d_hidden=64).to(cuda_device)

        # Same inputs
        torch.manual_seed(123)
        x = torch.randn(4, 128)

        # CPU forward
        head_cpu.eval()
        with torch.no_grad():
            output_cpu = head_cpu(x)

        # GPU forward
        head_gpu.eval()
        with torch.no_grad():
            output_gpu = head_gpu(x.to(cuda_device))

        # Compare
        assert torch.allclose(output_cpu, output_gpu.cpu(), atol=1e-5)

    def test_region_handler_cpu_gpu_consistency(self, cuda_device):
        """RegionHandler produces same results on CPU and GPU."""
        from src.models.components.region_handler import RegionHandler

        torch.manual_seed(42)
        handler_cpu = RegionHandler(d_model=64, n_regions=6)

        torch.manual_seed(42)
        handler_gpu = RegionHandler(d_model=64, n_regions=6).to(cuda_device)

        # Same inputs
        torch.manual_seed(123)
        x = torch.randn(4, 6, 31, 64)
        mask = torch.ones(4, 6, dtype=torch.bool)

        # CPU forward
        handler_cpu.eval()
        with torch.no_grad():
            pooled_cpu, context_cpu = handler_cpu(x, mask)

        # GPU forward
        handler_gpu.eval()
        with torch.no_grad():
            pooled_gpu, context_gpu = handler_gpu(
                x.to(cuda_device),
                mask.to(cuda_device)
            )

        # Compare
        assert torch.allclose(pooled_cpu, pooled_gpu.cpu(), atol=1e-5)
        assert torch.allclose(context_cpu, context_gpu.cpu(), atol=1e-5)


# ─────────────────────────────────────────────────────────────────────────────
# Mixed Precision Tests (Basic)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestMixedPrecisionBasic:
    """Basic mixed precision tests (more comprehensive tests in test_mixed_precision.py)."""

    def test_model_with_autocast(self, small_model_config, cuda_device, sample_inputs):
        """Model works with torch.amp.autocast."""
        from src.models.full_model import CognitiveResilienceModel

        model = CognitiveResilienceModel(**small_model_config, use_bayesian_head=False)
        model = model.to(cuda_device)
        model.eval()

        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                output = model(**sample_inputs)

        # Verify output exists and has correct shape
        assert output['mean'].shape == (2, 1)
        # Output might be float16 or float32 depending on autocast decisions
        assert output['mean'].device.type == "cuda"

    def test_fusion_layer_half_precision(self, cuda_device):
        """FusionLayer works with half precision inputs."""
        from src.models.fusion.fusion_layer import FusionLayer

        layer = FusionLayer(d_embed=64, d_fused=128, n_cell_types=31).to(cuda_device)
        layer = layer.half()  # Convert to float16

        pseudobulk = torch.randn(4, 31, 64, device=cuda_device, dtype=torch.float16)
        hgt = torch.randn(4, 31, 64, device=cuda_device, dtype=torch.float16)
        cell = torch.randn(4, 31, 64, device=cuda_device, dtype=torch.float16)

        output = layer(pseudobulk, hgt, cell)

        # Verify output
        assert output.dtype == torch.float16
        assert output.shape == (4, 31, 128)
        assert not torch.isnan(output).any()

    def test_deterministic_head_half_precision(self, cuda_device):
        """DeterministicPredictionHead works with half precision."""
        from src.models.heads.deterministic_head import DeterministicPredictionHead

        head = DeterministicPredictionHead(d_input=128, d_hidden=64).to(cuda_device)
        head = head.half()

        x = torch.randn(4, 128, device=cuda_device, dtype=torch.float16)

        output = head(x)

        # Verify output
        assert output.dtype == torch.float16
        assert output.shape == (4, 1)
        assert not torch.isnan(output).any()


# ─────────────────────────────────────────────────────────────────────────────
# HGT Encoder GPU Tests
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestHGTEncoderCUDA:
    """Test HGT Encoder on GPU."""

    def test_hgt_encoder_cuda(self, cuda_device):
        """HGTEncoder forward on GPU."""
        from src.models.branches.hgt_encoder import HGTEncoder
        from src.data.constants import CELL_TYPE_ORDER, ALL_EDGE_TYPES

        node_types = list(CELL_TYPE_ORDER)[:5]  # Use subset for speed
        edge_categories = list(ALL_EDGE_TYPES)

        encoder = HGTEncoder(
            d_input=64,
            d_hidden=64,
            d_output=64,
            n_heads=4,
            n_layers=1,
            dropout=0.0,
            edge_dim=1,
            node_types=node_types,
            edge_categories=edge_categories,
        ).to(cuda_device)

        # Create input dictionaries
        x_dict = {nt: torch.randn(1, 64, device=cuda_device) for nt in node_types}

        # Create simple edge structure
        edge_key = (node_types[0], edge_categories[0], node_types[1])
        edge_index_dict = {edge_key: torch.tensor([[0], [0]], device=cuda_device)}
        edge_attr_dict = {edge_key: torch.randn(1, 1, device=cuda_device)}

        out_dict, _ = encoder(x_dict, edge_index_dict, edge_attr_dict)

        # Verify outputs on CUDA
        for nt, out in out_dict.items():
            assert out.device.type == "cuda", f"Output for {nt} not on CUDA"
            assert out.shape == (1, 64)
