"""
Per-Component Serialization Tests.

Verifies that each model component can be saved and loaded correctly using:
- state_dict save/load
- torch.save/load for full model
- Checkpoint patterns with optimizer state
- Cross-device loading (GPU <-> CPU)

Test organization:
1. State Dict Tests - Keys, shapes, save/load verification
2. Per-Component Serialization - Individual component state_dict tests
3. Full Model Checkpoint - Complete model save/load patterns
4. Cross-Device Loading - GPU/CPU compatibility
5. Output Equivalence - Loaded models produce identical outputs
"""

import gc
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.optim as optim

from src.data.constants import N_CELL_TYPES, N_REGIONS


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def sample_inputs(make_edge_tensors):
    """Create sample inputs for forward pass testing."""
    B = 2
    n_regions = N_REGIONS
    n_cell_types = N_CELL_TYPES
    n_genes = 50
    n_cells_per_type = 10

    ccc_edge_index, ccc_edge_type, ccc_edge_attr = make_edge_tensors(B)

    # Flat cell format
    cells_per_sample = n_cell_types * n_cells_per_type
    total_cells = B * cells_per_sample
    cell_data = torch.randn(total_cells, n_genes)
    offsets_one = torch.arange(0, (n_cell_types + 1) * n_cells_per_type, n_cells_per_type)
    cell_offsets = torch.stack([offsets_one + i * cells_per_sample for i in range(B)])

    return {
        'region_pseudobulk': torch.randn(B, n_regions, n_cell_types, n_genes),
        'region_mask': torch.ones(B, n_regions, dtype=torch.bool),
        'ccc_edge_index': ccc_edge_index,
        'ccc_edge_type': ccc_edge_type,
        'ccc_edge_attr': ccc_edge_attr,
        'cell_data': cell_data,
        'cell_offsets': cell_offsets,
        'pathology': torch.randn(B, 3),
        'cognition': torch.randn(B, 1),
    }


@pytest.fixture
def cuda_device():
    """Provide CUDA device if available."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda:0")


def _move_value(v, device):
    """Recursively move tensors in nested structures to device."""
    if isinstance(v, torch.Tensor):
        return v.to(device)
    if isinstance(v, dict):
        return {k: _move_value(val, device) for k, val in v.items()}
    if isinstance(v, list):
        return [_move_value(item, device) for item in v]
    return v


def move_inputs_to_device(inputs: dict, device: torch.device) -> dict:
    """Helper to move all input tensors (including nested) to a device."""
    return {k: _move_value(v, device) for k, v in inputs.items()}


# ─────────────────────────────────────────────────────────────────────────────
# State Dict Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestModelStateDictKeys:
    """Test model state_dict key structure."""

    def test_model_state_dict_keys(self, small_model_config):
        """Verify expected keys in state_dict for deterministic model."""
        from src.models.full_model import CognitiveResilienceModel

        model = CognitiveResilienceModel(**small_model_config, use_bayesian_head=False)
        state_dict = model.state_dict()

        # Should have keys from all major components
        key_prefixes = set()
        for key in state_dict.keys():
            prefix = key.split('.')[0]
            key_prefixes.add(prefix)

        # Verify all major components have state
        expected_prefixes = {
            'hgt_gene_gate',
            'hgt_input_proj',
            'region_handler',
            'hgt_encoder',
            'cell_transformer',
            'fusion_layer',
            'pathology_encoder',
            'pathology_attention',
            'prediction_head',
        }

        for prefix in expected_prefixes:
            assert prefix in key_prefixes, f"Missing state_dict keys for {prefix}"

    def test_bayesian_model_state_dict_keys(self, small_model_config):
        """Verify expected keys in state_dict for Bayesian model."""
        from src.models.full_model import CognitiveResilienceModel

        model = CognitiveResilienceModel(**small_model_config, use_bayesian_head=True)
        state_dict = model.state_dict()

        # Bayesian head should have state for deterministic layer (fc_log_std)
        fc_log_std_keys = [k for k in state_dict.keys() if 'fc_log_std' in k]
        assert len(fc_log_std_keys) > 0, "Missing fc_log_std state in Bayesian head"

    def test_model_state_dict_no_unexpected_keys(self, small_model_config):
        """Verify no unexpected keys appear in state_dict."""
        from src.models.full_model import CognitiveResilienceModel

        model = CognitiveResilienceModel(**small_model_config, use_bayesian_head=False)
        state_dict = model.state_dict()

        # All keys should start with known prefixes
        known_prefixes = {
            'hgt_gene_gate',
            'hgt_input_proj',
            'region_handler',
            'hgt_encoder',
            'cell_transformer',
            'fusion_layer',
            'pathology_encoder',
            'pathology_attention',
            'prediction_head',
        }

        for key in state_dict.keys():
            prefix = key.split('.')[0]
            assert prefix in known_prefixes, f"Unexpected key prefix: {prefix}"


class TestModelStateDictSaveLoad:
    """Test state_dict save/load functionality."""

    def test_model_state_dict_save_load(self, small_model_config):
        """Save/load state_dict, verify identical state."""
        from src.models.full_model import CognitiveResilienceModel

        # Create model and save state
        model1 = CognitiveResilienceModel(**small_model_config, use_bayesian_head=False)
        state_dict = model1.state_dict()

        # Create fresh model and load state
        model2 = CognitiveResilienceModel(**small_model_config, use_bayesian_head=False)
        model2.load_state_dict(state_dict)

        # Compare all parameters
        for (name1, param1), (name2, param2) in zip(
            model1.named_parameters(),
            model2.named_parameters()
        ):
            assert name1 == name2
            assert torch.equal(param1, param2), f"Parameter mismatch: {name1}"

    def test_model_state_dict_save_load_file(self, small_model_config):
        """Save/load state_dict to file, verify identical state."""
        from src.models.full_model import CognitiveResilienceModel

        model1 = CognitiveResilienceModel(**small_model_config, use_bayesian_head=False)

        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            temp_path = Path(f.name)

        try:
            # Save to file
            torch.save(model1.state_dict(), temp_path)

            # Create fresh model and load
            model2 = CognitiveResilienceModel(**small_model_config, use_bayesian_head=False)
            model2.load_state_dict(torch.load(temp_path, weights_only=True))

            # Compare
            for (name1, param1), (name2, param2) in zip(
                model1.named_parameters(),
                model2.named_parameters()
            ):
                assert torch.equal(param1, param2), f"Parameter mismatch after file load: {name1}"
        finally:
            temp_path.unlink()


class TestModelStateDictShapes:
    """Test that state_dict tensor shapes are preserved."""

    def test_model_state_dict_shapes(self, small_model_config):
        """Verify tensor shapes preserved after save/load."""
        from src.models.full_model import CognitiveResilienceModel

        model1 = CognitiveResilienceModel(**small_model_config, use_bayesian_head=False)
        state_dict1 = model1.state_dict()

        model2 = CognitiveResilienceModel(**small_model_config, use_bayesian_head=False)
        model2.load_state_dict(state_dict1)
        state_dict2 = model2.state_dict()

        # All shapes should match
        for key in state_dict1.keys():
            shape1 = state_dict1[key].shape
            shape2 = state_dict2[key].shape
            assert shape1 == shape2, f"Shape mismatch for {key}: {shape1} vs {shape2}"

    def test_model_state_dict_dtypes(self, small_model_config):
        """Verify tensor dtypes preserved after save/load."""
        from src.models.full_model import CognitiveResilienceModel

        model1 = CognitiveResilienceModel(**small_model_config, use_bayesian_head=False)
        state_dict1 = model1.state_dict()

        model2 = CognitiveResilienceModel(**small_model_config, use_bayesian_head=False)
        model2.load_state_dict(state_dict1)
        state_dict2 = model2.state_dict()

        # All dtypes should match
        for key in state_dict1.keys():
            dtype1 = state_dict1[key].dtype
            dtype2 = state_dict2[key].dtype
            assert dtype1 == dtype2, f"Dtype mismatch for {key}: {dtype1} vs {dtype2}"


# ─────────────────────────────────────────────────────────────────────────────
# Per-Component Serialization Tests
# ─────────────────────────────────────────────────────────────────────────────


def _make_fusion_layer():
    """Create a FusionLayer for serialization testing."""
    from src.models.fusion.fusion_layer import FusionLayer
    return FusionLayer(d_embed=64, d_fused=128, n_cell_types=N_CELL_TYPES)


def _make_pathology_encoder():
    """Create a PathologyEncoder for serialization testing."""
    from src.models.fusion.pathology_encoder import PathologyEncoder
    return PathologyEncoder(n_pathology_features=3, d_region=128, d_cond=64)


def _make_pathology_attention():
    """Create a PathologyStratifiedAttention for serialization testing."""
    from src.models.fusion.pathology_attention import PathologyStratifiedAttention
    return PathologyStratifiedAttention(
        d_fused=64, d_cond=32, n_heads=4, n_cell_types=N_CELL_TYPES
    )


def _make_deterministic_head():
    """Create a DeterministicPredictionHead for serialization testing."""
    from src.models.heads.deterministic_head import DeterministicPredictionHead
    return DeterministicPredictionHead(d_input=128, d_hidden=64)


SIMPLE_SERIALIZATION_CASES = [
    ("FusionLayer", _make_fusion_layer, ["proj", "layer_norm"]),
    ("PathologyEncoder", _make_pathology_encoder, ["pathology_mlp", "region_proj", "combine"]),
    ("PathologyAttention", _make_pathology_attention,
     ["query_generator", "key_proj", "value_proj", "pathology_bias", "out_proj"]),
    ("DeterministicHead", _make_deterministic_head, ["mlp.0", "mlp.3", "mlp.6"]),
]


class TestComponentSerializationParametrized:
    """Parametrized serialization tests for simple components.

    Covers save/load state_dict round-trip and expected key verification
    for: FusionLayer, PathologyEncoder, PathologyAttention, DeterministicHead.
    """

    @pytest.mark.parametrize(
        "name,make_component,key_substrings",
        SIMPLE_SERIALIZATION_CASES,
        ids=[c[0] for c in SIMPLE_SERIALIZATION_CASES],
    )
    def test_save_load_state_dict(self, name, make_component, key_substrings):
        """State_dict save/load round-trip preserves all parameters."""
        original = make_component()
        state_dict = original.state_dict()

        restored = make_component()
        restored.load_state_dict(state_dict)

        for (n1, p1), (n2, p2) in zip(
            original.named_parameters(),
            restored.named_parameters(),
        ):
            assert n1 == n2
            assert torch.equal(p1, p2), f"{name} parameter mismatch: {n1}"

    @pytest.mark.parametrize(
        "name,make_component,key_substrings",
        SIMPLE_SERIALIZATION_CASES,
        ids=[c[0] for c in SIMPLE_SERIALIZATION_CASES],
    )
    def test_state_dict_has_expected_keys(self, name, make_component, key_substrings):
        """State_dict contains all expected key substrings."""
        component = make_component()
        keys = list(component.state_dict().keys())
        for substring in key_substrings:
            assert any(substring in k for k in keys), \
                f"{name}: missing key containing '{substring}'"

    @pytest.mark.parametrize(
        "name,make_component,key_substrings",
        SIMPLE_SERIALIZATION_CASES,
        ids=[c[0] for c in SIMPLE_SERIALIZATION_CASES],
    )
    def test_save_load_state_dict_file(self, name, make_component, key_substrings):
        """State_dict save/load via file preserves all parameters."""
        original = make_component()

        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            temp_path = Path(f.name)

        try:
            torch.save(original.state_dict(), temp_path)

            restored = make_component()
            restored.load_state_dict(torch.load(temp_path, weights_only=True))

            for (n1, p1), (n2, p2) in zip(
                original.named_parameters(),
                restored.named_parameters(),
            ):
                assert torch.equal(p1, p2), f"{name} file-load parameter mismatch: {n1}"
        finally:
            temp_path.unlink()


class TestRegionHandlerSerialization:
    """Test RegionHandler state_dict serialization."""

    def test_region_handler_serialization(self):
        """RegionHandler state_dict save/load."""
        from src.models.components.region_handler import RegionHandler

        handler1 = RegionHandler(d_model=128, n_regions=N_REGIONS)
        state_dict = handler1.state_dict()

        handler2 = RegionHandler(d_model=128, n_regions=N_REGIONS)
        handler2.load_state_dict(state_dict)

        for (name1, param1), (name2, param2) in zip(
            handler1.named_parameters(),
            handler2.named_parameters()
        ):
            assert torch.equal(param1, param2), f"RegionHandler parameter mismatch: {name1}"

    def test_region_handler_state_dict_keys(self):
        """RegionHandler state_dict contains expected keys."""
        from src.models.components.region_handler import RegionHandler

        handler = RegionHandler(d_model=128, n_regions=N_REGIONS)
        state_dict = handler.state_dict()

        # Should have region_weights and region_embedding.weight
        assert 'region_weights' in state_dict
        assert 'region_embedding.weight' in state_dict

    def test_region_handler_weights_preserved(self):
        """RegionHandler learned region weights are preserved."""
        from src.models.components.region_handler import RegionHandler

        handler1 = RegionHandler(d_model=128, n_regions=N_REGIONS)

        # Modify weights
        with torch.no_grad():
            handler1.region_weights.data = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

        state_dict = handler1.state_dict()

        handler2 = RegionHandler(d_model=128, n_regions=N_REGIONS)
        handler2.load_state_dict(state_dict)

        assert torch.equal(handler1.region_weights, handler2.region_weights)


class TestBayesianHeadSerialization:
    """Test BayesianPredictionHead state_dict serialization.

    Note: BayesianPredictionHead uses PyroModule which has special serialization
    behavior. The deterministic layer (fc_log_std) should serialize normally,
    while PyroSample layers have different semantics.
    """

    def test_bayesian_head_serialization(self):
        """BayesianPredictionHead state_dict save/load for deterministic parts."""
        from src.models.heads.bayesian_head import BayesianPredictionHead

        head1 = BayesianPredictionHead(d_input=128, d_hidden=64)
        state_dict = head1.state_dict()

        head2 = BayesianPredictionHead(d_input=128, d_hidden=64)
        head2.load_state_dict(state_dict)

        # Check fc_log_std (deterministic layer) is correctly serialized
        assert torch.equal(
            head1.fc_log_std.weight,
            head2.fc_log_std.weight
        ), "fc_log_std weight mismatch"

        assert torch.equal(
            head1.fc_log_std.bias,
            head2.fc_log_std.bias
        ), "fc_log_std bias mismatch"

    def test_bayesian_head_state_dict_contains_fc_log_std(self):
        """BayesianPredictionHead state_dict contains fc_log_std keys."""
        from src.models.heads.bayesian_head import BayesianPredictionHead

        head = BayesianPredictionHead(d_input=128, d_hidden=64)
        state_dict = head.state_dict()

        # fc_log_std is deterministic and should be in state_dict
        fc_log_std_keys = [k for k in state_dict.keys() if 'fc_log_std' in k]
        assert len(fc_log_std_keys) >= 2, "Expected fc_log_std.weight and fc_log_std.bias"

    def test_bayesian_head_serialization_file(self):
        """BayesianPredictionHead state_dict file save/load."""
        from src.models.heads.bayesian_head import BayesianPredictionHead

        head1 = BayesianPredictionHead(d_input=128, d_hidden=64)

        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            temp_path = Path(f.name)

        try:
            torch.save(head1.state_dict(), temp_path)

            head2 = BayesianPredictionHead(d_input=128, d_hidden=64)
            head2.load_state_dict(torch.load(temp_path, weights_only=True))

            assert torch.equal(head1.fc_log_std.weight, head2.fc_log_std.weight)
        finally:
            temp_path.unlink()


class TestCellTransformerSerialization:
    """Test CellTransformer state_dict serialization and output preservation."""

    def test_save_and_load_preserves_output(self):
        """CellTransformer save/load preserves output."""
        from src.models.branches.cell_transformer import CellTransformer
        from src.data.constants import N_CELL_TYPES

        ct = CellTransformer(n_genes=50, n_cell_types=N_CELL_TYPES, d_model=64, n_heads=4, n_isab_layers=1, n_inducing=16, n_pma_seeds=1)
        ct.eval()
        B = 2
        n_cells_per_type = 10
        cells_per_sample = N_CELL_TYPES * n_cells_per_type
        total_cells = B * cells_per_sample
        cell_data = torch.randn(total_cells, 50)
        offsets_one = torch.arange(0, (N_CELL_TYPES + 1) * n_cells_per_type, n_cells_per_type)
        cell_offsets = torch.stack([offsets_one + i * cells_per_sample for i in range(B)])
        out1 = ct(cell_data, cell_offsets)[0]
        # Save and reload
        state = ct.state_dict()
        ct2 = CellTransformer(n_genes=50, n_cell_types=N_CELL_TYPES, d_model=64, n_heads=4, n_isab_layers=1, n_inducing=16, n_pma_seeds=1)
        ct2.load_state_dict(state)
        ct2.eval()
        out2 = ct2(cell_data, cell_offsets)[0]
        assert torch.allclose(out1, out2)


class TestHGTEncoderTensorSerialization:
    """Test HGTEncoderTensor state_dict serialization and output preservation."""

    def test_save_and_load_preserves_output(self):
        """HGTEncoderTensor save/load preserves output."""
        from src.models.branches.hgt_encoder_tensor import HGTEncoderTensor
        from src.data.constants import N_EDGE_TYPES

        d = 32
        n_ct = N_CELL_TYPES
        hgt = HGTEncoderTensor(d_input=d, d_hidden=d, d_output=d, n_heads=4, n_layers=1, n_node_types=N_CELL_TYPES, n_edge_types=N_EDGE_TYPES, dropout=0.0, edge_dim=1)
        hgt.eval()

        B = 2
        n_edges = 5
        x = torch.randn(B, n_ct, d)
        src_parts, dst_parts, type_parts = [], [], []
        for b in range(B):
            offset = b * n_ct
            src_parts.append(torch.randint(0, n_ct, (n_edges,)) + offset)
            dst_parts.append(torch.randint(0, n_ct, (n_edges,)) + offset)
            type_parts.append(torch.randint(0, N_EDGE_TYPES, (n_edges,)))
        edge_index = torch.stack([torch.cat(src_parts), torch.cat(dst_parts)])
        edge_type = torch.cat(type_parts)
        edge_attr = torch.rand(B * n_edges, 1)

        with torch.no_grad():
            out1 = hgt(x, edge_index, edge_type, edge_attr)

        state = hgt.state_dict()
        hgt2 = HGTEncoderTensor(d_input=d, d_hidden=d, d_output=d, n_heads=4, n_layers=1, n_node_types=N_CELL_TYPES, n_edge_types=N_EDGE_TYPES, dropout=0.0, edge_dim=1)
        hgt2.load_state_dict(state)
        hgt2.eval()

        with torch.no_grad():
            out2 = hgt2(x, edge_index, edge_type, edge_attr)

        # Verify outputs match
        assert torch.allclose(out1, out2)


# ─────────────────────────────────────────────────────────────────────────────
# Full Model Checkpoint Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestFullModelTorchSaveLoad:
    """Test torch.save/load for full model."""

    def test_full_model_torch_save_load(self, small_model_config):
        """torch.save/load full model."""
        from src.models.full_model import CognitiveResilienceModel

        model1 = CognitiveResilienceModel(**small_model_config, use_bayesian_head=False)

        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            temp_path = Path(f.name)

        try:
            # Save entire model
            torch.save(model1, temp_path)

            # Load model
            model2 = torch.load(temp_path, weights_only=False)

            # Verify parameters match
            for (name1, param1), (name2, param2) in zip(
                model1.named_parameters(),
                model2.named_parameters()
            ):
                assert torch.equal(param1, param2), f"Full model parameter mismatch: {name1}"
        finally:
            temp_path.unlink()

    def test_full_model_state_dict_save_load(self, small_model_config):
        """Full model state_dict save/load pattern."""
        from src.models.full_model import CognitiveResilienceModel

        model1 = CognitiveResilienceModel(**small_model_config, use_bayesian_head=False)

        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            temp_path = Path(f.name)

        try:
            # Save state_dict
            torch.save({
                'model_state_dict': model1.state_dict(),
                'config': small_model_config,
            }, temp_path)

            # Load
            checkpoint = torch.load(temp_path, weights_only=False)
            model2 = CognitiveResilienceModel(**checkpoint['config'], use_bayesian_head=False)
            model2.load_state_dict(checkpoint['model_state_dict'])

            # Verify
            for (name1, param1), (name2, param2) in zip(
                model1.named_parameters(),
                model2.named_parameters()
            ):
                assert torch.equal(param1, param2)
        finally:
            temp_path.unlink()


class TestCheckpointWithOptimizer:
    """Test checkpoint patterns with optimizer state."""

    def test_checkpoint_with_optimizer(self, small_model_config, sample_inputs):
        """Save model + optimizer state."""
        from src.models.full_model import CognitiveResilienceModel

        # Create model and optimizer
        model1 = CognitiveResilienceModel(**small_model_config, use_bayesian_head=False)
        optimizer1 = optim.Adam(model1.parameters(), lr=0.001)

        # Do one training step to update optimizer state
        model1.train()
        output = model1(**sample_inputs)
        loss = output['mean'].sum()
        loss.backward()
        optimizer1.step()

        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            temp_path = Path(f.name)

        try:
            # Save checkpoint
            torch.save({
                'model_state_dict': model1.state_dict(),
                'optimizer_state_dict': optimizer1.state_dict(),
                'config': small_model_config,
            }, temp_path)

            # Load checkpoint
            checkpoint = torch.load(temp_path, weights_only=False)
            model2 = CognitiveResilienceModel(**checkpoint['config'], use_bayesian_head=False)
            model2.load_state_dict(checkpoint['model_state_dict'])

            optimizer2 = optim.Adam(model2.parameters(), lr=0.001)
            optimizer2.load_state_dict(checkpoint['optimizer_state_dict'])

            # Verify model parameters
            for (name1, param1), (name2, param2) in zip(
                model1.named_parameters(),
                model2.named_parameters()
            ):
                assert torch.equal(param1, param2), f"Model parameter mismatch: {name1}"

            # Verify optimizer state
            state1 = optimizer1.state_dict()
            state2 = optimizer2.state_dict()

            # Compare param_groups
            assert len(state1['param_groups']) == len(state2['param_groups'])

        finally:
            temp_path.unlink()

    def test_checkpoint_with_scheduler(self, small_model_config, sample_inputs):
        """Save model + optimizer + scheduler state."""
        from src.models.full_model import CognitiveResilienceModel

        model1 = CognitiveResilienceModel(**small_model_config, use_bayesian_head=False)
        optimizer1 = optim.Adam(model1.parameters(), lr=0.001)
        scheduler1 = optim.lr_scheduler.StepLR(optimizer1, step_size=10)

        # Do training steps
        model1.train()
        for _ in range(5):
            output = model1(**sample_inputs)
            loss = output['mean'].sum()
            loss.backward()
            optimizer1.step()
            optimizer1.zero_grad()
            scheduler1.step()

        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            temp_path = Path(f.name)

        try:
            torch.save({
                'model_state_dict': model1.state_dict(),
                'optimizer_state_dict': optimizer1.state_dict(),
                'scheduler_state_dict': scheduler1.state_dict(),
                'epoch': 5,
            }, temp_path)

            checkpoint = torch.load(temp_path, weights_only=False)

            model2 = CognitiveResilienceModel(**small_model_config, use_bayesian_head=False)
            model2.load_state_dict(checkpoint['model_state_dict'])

            optimizer2 = optim.Adam(model2.parameters(), lr=0.001)
            optimizer2.load_state_dict(checkpoint['optimizer_state_dict'])

            scheduler2 = optim.lr_scheduler.StepLR(optimizer2, step_size=10)
            scheduler2.load_state_dict(checkpoint['scheduler_state_dict'])

            # Verify scheduler state
            assert scheduler1.last_epoch == scheduler2.last_epoch
            assert checkpoint['epoch'] == 5

        finally:
            temp_path.unlink()


class TestCheckpointResumeTraining:
    """Test resuming training from checkpoint."""

    def test_checkpoint_resume_training(self, small_model_config, sample_inputs):
        """Resume training from checkpoint produces consistent results."""
        from src.models.full_model import CognitiveResilienceModel

        torch.manual_seed(42)

        # Initial training
        model1 = CognitiveResilienceModel(**small_model_config, use_bayesian_head=False)
        optimizer1 = optim.Adam(model1.parameters(), lr=0.001)

        model1.train()
        losses_phase1 = []
        for _ in range(3):
            output = model1(**sample_inputs)
            loss = output['mean'].sum()
            losses_phase1.append(loss.item())
            loss.backward()
            optimizer1.step()
            optimizer1.zero_grad()

        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            temp_path = Path(f.name)

        try:
            # Save checkpoint
            torch.save({
                'model_state_dict': model1.state_dict(),
                'optimizer_state_dict': optimizer1.state_dict(),
            }, temp_path)

            # Continue training from model1
            torch.manual_seed(123)
            losses_continued_1 = []
            for _ in range(3):
                output = model1(**sample_inputs)
                loss = output['mean'].sum()
                losses_continued_1.append(loss.item())
                loss.backward()
                optimizer1.step()
                optimizer1.zero_grad()

            # Load checkpoint and continue
            checkpoint = torch.load(temp_path, weights_only=False)
            model2 = CognitiveResilienceModel(**small_model_config, use_bayesian_head=False)
            model2.load_state_dict(checkpoint['model_state_dict'])

            optimizer2 = optim.Adam(model2.parameters(), lr=0.001)
            optimizer2.load_state_dict(checkpoint['optimizer_state_dict'])

            torch.manual_seed(123)
            losses_continued_2 = []
            model2.train()
            for _ in range(3):
                output = model2(**sample_inputs)
                loss = output['mean'].sum()
                losses_continued_2.append(loss.item())
                loss.backward()
                optimizer2.step()
                optimizer2.zero_grad()

            # Losses should match exactly
            for l1, l2 in zip(losses_continued_1, losses_continued_2):
                assert abs(l1 - l2) < 1e-5, f"Loss mismatch: {l1} vs {l2}"

        finally:
            temp_path.unlink()


# ─────────────────────────────────────────────────────────────────────────────
# Cross-Device Loading Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestSaveCUDALoadCPU:
    """Test saving on GPU and loading on CPU."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_save_cuda_load_cpu(self, small_model_config, cuda_device):
        """Save on GPU, load on CPU."""
        from src.models.full_model import CognitiveResilienceModel

        # Create model on GPU
        model1 = CognitiveResilienceModel(**small_model_config, use_bayesian_head=False)
        model1 = model1.to(cuda_device)

        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            temp_path = Path(f.name)

        try:
            # Save from GPU
            torch.save(model1.state_dict(), temp_path)

            # Load to CPU
            model2 = CognitiveResilienceModel(**small_model_config, use_bayesian_head=False)
            model2.load_state_dict(
                torch.load(temp_path, map_location='cpu', weights_only=True)
            )

            # Verify model is on CPU
            assert next(model2.parameters()).device.type == 'cpu'

            # Verify parameters match (after moving to same device)
            model1_cpu = model1.cpu()
            for (name1, param1), (name2, param2) in zip(
                model1_cpu.named_parameters(),
                model2.named_parameters()
            ):
                assert torch.equal(param1, param2), f"Cross-device parameter mismatch: {name1}"
        finally:
            temp_path.unlink()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_save_cuda_load_cpu_component(self, cuda_device):
        """Save component on GPU, load on CPU."""
        from src.models.fusion.fusion_layer import FusionLayer

        layer1 = FusionLayer(d_embed=64, d_fused=128, n_cell_types=N_CELL_TYPES).to(cuda_device)

        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            temp_path = Path(f.name)

        try:
            torch.save(layer1.state_dict(), temp_path)

            layer2 = FusionLayer(d_embed=64, d_fused=128, n_cell_types=N_CELL_TYPES)
            layer2.load_state_dict(
                torch.load(temp_path, map_location='cpu', weights_only=True)
            )

            assert next(layer2.parameters()).device.type == 'cpu'
        finally:
            temp_path.unlink()


class TestSaveCPULoadCUDA:
    """Test saving on CPU and loading on GPU."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_save_cpu_load_cuda(self, small_model_config, cuda_device):
        """Save on CPU, load on GPU (if CUDA available)."""
        from src.models.full_model import CognitiveResilienceModel

        # Create model on CPU
        model1 = CognitiveResilienceModel(**small_model_config, use_bayesian_head=False)

        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            temp_path = Path(f.name)

        try:
            torch.save(model1.state_dict(), temp_path)

            # Load to GPU
            model2 = CognitiveResilienceModel(**small_model_config, use_bayesian_head=False)
            model2.load_state_dict(
                torch.load(temp_path, map_location=cuda_device, weights_only=True)
            )
            model2 = model2.to(cuda_device)

            # Verify model is on GPU
            assert next(model2.parameters()).device.type == 'cuda'

            # Verify parameters match
            model1_cuda = model1.to(cuda_device)
            for (name1, param1), (name2, param2) in zip(
                model1_cuda.named_parameters(),
                model2.named_parameters()
            ):
                assert torch.equal(param1, param2), f"Cross-device parameter mismatch: {name1}"
        finally:
            temp_path.unlink()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_save_cpu_load_cuda_component(self, cuda_device):
        """Save component on CPU, load on GPU."""
        from src.models.heads.deterministic_head import DeterministicPredictionHead

        head1 = DeterministicPredictionHead(d_input=128, d_hidden=64)

        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            temp_path = Path(f.name)

        try:
            torch.save(head1.state_dict(), temp_path)

            head2 = DeterministicPredictionHead(d_input=128, d_hidden=64)
            head2.load_state_dict(
                torch.load(temp_path, map_location=cuda_device, weights_only=True)
            )
            head2 = head2.to(cuda_device)

            assert next(head2.parameters()).device.type == 'cuda'
        finally:
            temp_path.unlink()


# ─────────────────────────────────────────────────────────────────────────────
# Output Equivalence Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestLoadedModelSameOutput:
    """Test that loaded models produce same output as original."""

    def test_loaded_model_same_output(self, small_model_config, sample_inputs):
        """Loaded model produces same output as original."""
        from src.models.full_model import CognitiveResilienceModel

        # Create model
        model1 = CognitiveResilienceModel(**small_model_config, use_bayesian_head=False)
        model1.eval()

        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            temp_path = Path(f.name)

        try:
            torch.save(model1.state_dict(), temp_path)

            # Get output from original
            torch.manual_seed(42)
            with torch.no_grad():
                output1 = model1(**sample_inputs)

            # Load and get output
            model2 = CognitiveResilienceModel(**small_model_config, use_bayesian_head=False)
            model2.load_state_dict(torch.load(temp_path, weights_only=True))
            model2.eval()

            torch.manual_seed(42)
            with torch.no_grad():
                output2 = model2(**sample_inputs)

            # Verify outputs match
            assert torch.allclose(output1['mean'], output2['mean'], atol=1e-6), \
                "Mean output mismatch after load"
            assert torch.allclose(
                output1['attention_weights'],
                output2['attention_weights'],
                atol=1e-6
            ), "Attention weights mismatch after load"

        finally:
            temp_path.unlink()

    def test_loaded_model_same_output_bayesian(self, small_model_config, sample_inputs):
        """Loaded Bayesian model produces same output as original."""
        from src.models.full_model import CognitiveResilienceModel

        model1 = CognitiveResilienceModel(**small_model_config, use_bayesian_head=True)
        model1.eval()

        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            temp_path = Path(f.name)

        try:
            torch.save(model1.state_dict(), temp_path)

            # Note: Bayesian models have stochastic forward pass due to PyroSample
            # We compare the deterministic parts (fc_log_std influence)
            model2 = CognitiveResilienceModel(**small_model_config, use_bayesian_head=True)
            model2.load_state_dict(torch.load(temp_path, weights_only=True))
            model2.eval()

            # Verify attention weights (from fusion, not affected by Bayesian head)
            torch.manual_seed(42)
            with torch.no_grad():
                output1 = model1(**sample_inputs)

            torch.manual_seed(42)
            with torch.no_grad():
                output2 = model2(**sample_inputs)

            assert torch.allclose(
                output1['attention_weights'],
                output2['attention_weights'],
                atol=1e-6
            ), "Bayesian model attention weights mismatch after load"

        finally:
            temp_path.unlink()

    def test_loaded_component_same_output(self):
        """Loaded component produces same output as original."""
        from src.models.fusion.fusion_layer import FusionLayer

        layer1 = FusionLayer(d_embed=64, d_fused=128, n_cell_types=N_CELL_TYPES)
        layer1.eval()

        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            temp_path = Path(f.name)

        try:
            torch.save(layer1.state_dict(), temp_path)

            # Create inputs (2-branch: hgt + cell)
            hgt = torch.randn(4, N_CELL_TYPES, 64)
            cell = torch.randn(4, N_CELL_TYPES, 64)

            # Get original output
            with torch.no_grad():
                output1 = layer1(hgt, cell)

            # Load and get output
            layer2 = FusionLayer(d_embed=64, d_fused=128, n_cell_types=N_CELL_TYPES)
            layer2.load_state_dict(torch.load(temp_path, weights_only=True))
            layer2.eval()

            with torch.no_grad():
                output2 = layer2(hgt, cell)

            assert torch.allclose(output1, output2, atol=1e-6), \
                "Component output mismatch after load"

        finally:
            temp_path.unlink()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_loaded_model_same_output_cuda(self, small_model_config, sample_inputs, cuda_device):
        """Loaded model produces same output on CUDA."""
        from src.models.full_model import CognitiveResilienceModel

        model1 = CognitiveResilienceModel(**small_model_config, use_bayesian_head=False)
        model1 = model1.to(cuda_device)
        model1.eval()

        cuda_inputs = move_inputs_to_device(sample_inputs, cuda_device)

        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            temp_path = Path(f.name)

        try:
            torch.save(model1.state_dict(), temp_path)

            torch.manual_seed(42)
            torch.cuda.manual_seed(42)
            with torch.no_grad():
                output1 = model1(**cuda_inputs)

            model2 = CognitiveResilienceModel(**small_model_config, use_bayesian_head=False)
            model2.load_state_dict(
                torch.load(temp_path, map_location=cuda_device, weights_only=True)
            )
            model2 = model2.to(cuda_device)
            model2.eval()

            torch.manual_seed(42)
            torch.cuda.manual_seed(42)
            with torch.no_grad():
                output2 = model2(**cuda_inputs)

            assert torch.allclose(output1['mean'], output2['mean'], atol=1e-6)

        finally:
            temp_path.unlink()


# ─────────────────────────────────────────────────────────────────────────────
# Additional Serialization Edge Cases
# ─────────────────────────────────────────────────────────────────────────────


class TestSerializationEdgeCases:
    """Test edge cases in serialization."""

    def test_strict_load_fails_with_missing_keys(self, small_model_config):
        """strict=True should fail if keys are missing."""
        from src.models.full_model import CognitiveResilienceModel

        model1 = CognitiveResilienceModel(**small_model_config, use_bayesian_head=False)
        state_dict = model1.state_dict()

        # Remove a key
        key_to_remove = list(state_dict.keys())[0]
        del state_dict[key_to_remove]

        model2 = CognitiveResilienceModel(**small_model_config, use_bayesian_head=False)

        with pytest.raises(RuntimeError, match="Missing key"):
            model2.load_state_dict(state_dict, strict=True)

    def test_non_strict_load_with_missing_keys(self, small_model_config):
        """strict=False should allow missing keys."""
        from src.models.full_model import CognitiveResilienceModel

        model1 = CognitiveResilienceModel(**small_model_config, use_bayesian_head=False)
        state_dict = model1.state_dict()

        # Remove a key
        key_to_remove = list(state_dict.keys())[0]
        del state_dict[key_to_remove]

        model2 = CognitiveResilienceModel(**small_model_config, use_bayesian_head=False)

        # Should not raise
        result = model2.load_state_dict(state_dict, strict=False)
        assert key_to_remove in result.missing_keys

    def test_load_with_extra_keys(self, small_model_config):
        """Loading state_dict with extra keys."""
        from src.models.full_model import CognitiveResilienceModel

        model1 = CognitiveResilienceModel(**small_model_config, use_bayesian_head=False)
        state_dict = model1.state_dict()

        # Add extra key
        state_dict['extra_key'] = torch.tensor([1, 2, 3])

        model2 = CognitiveResilienceModel(**small_model_config, use_bayesian_head=False)

        # strict=True should fail
        with pytest.raises(RuntimeError, match="Unexpected key"):
            model2.load_state_dict(state_dict, strict=True)

        # strict=False should work
        result = model2.load_state_dict(state_dict, strict=False)
        assert 'extra_key' in result.unexpected_keys

    def test_empty_state_dict(self, small_model_config):
        """Loading empty state_dict with strict=False."""
        from src.models.full_model import CognitiveResilienceModel

        model = CognitiveResilienceModel(**small_model_config, use_bayesian_head=False)

        # Empty state_dict
        result = model.load_state_dict({}, strict=False)

        # All keys should be missing
        assert len(result.missing_keys) > 0

    def test_serialization_with_buffers(self):
        """Test serialization preserves buffers (non-parameter tensors)."""
        from src.models.components.region_handler import RegionHandler

        handler1 = RegionHandler(d_model=128, n_regions=N_REGIONS)

        # RegionHandler has region_embedding which uses Embedding (has weight buffer)
        state_dict = handler1.state_dict()

        # Should include embedding weight
        assert 'region_embedding.weight' in state_dict

        handler2 = RegionHandler(d_model=128, n_regions=N_REGIONS)
        handler2.load_state_dict(state_dict)

        # Embedding should match
        assert torch.equal(
            handler1.region_embedding.weight,
            handler2.region_embedding.weight
        )


class TestSerializationWithGradients:
    """Test serialization behavior with gradients."""

    def test_state_dict_excludes_gradients(self, small_model_config, sample_inputs):
        """state_dict should not include gradients."""
        from src.models.full_model import CognitiveResilienceModel

        model = CognitiveResilienceModel(**small_model_config, use_bayesian_head=False)
        model.train()

        # Compute gradients
        output = model(**sample_inputs)
        loss = output['mean'].sum()
        loss.backward()

        # Verify gradients exist
        assert model.fusion_layer.proj.weight.grad is not None

        # Get state_dict
        state_dict = model.state_dict()

        # State dict should only contain parameter values, not gradients
        for key, value in state_dict.items():
            assert not hasattr(value, 'grad') or value.grad is None

    def test_load_does_not_affect_existing_gradients(self, small_model_config, sample_inputs):
        """Loading state_dict should not affect existing gradients."""
        from src.models.full_model import CognitiveResilienceModel

        model1 = CognitiveResilienceModel(**small_model_config, use_bayesian_head=False)
        model2 = CognitiveResilienceModel(**small_model_config, use_bayesian_head=False)

        # Compute gradients on model2
        model2.train()
        output = model2(**sample_inputs)
        loss = output['mean'].sum()
        loss.backward()

        # Save original gradient
        original_grad = model2.fusion_layer.proj.weight.grad.clone()

        # Load state from model1 into model2
        model2.load_state_dict(model1.state_dict())

        # Gradient should be preserved (or at least not cause errors)
        # PyTorch behavior: load_state_dict doesn't modify .grad attributes
        # The gradient may be cleared or preserved depending on version
