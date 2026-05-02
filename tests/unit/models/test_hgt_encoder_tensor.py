"""Unit tests for HGTEncoderTensor — batched tensor-native HGT encoder."""

import pytest
import torch

from src.data.constants import N_CELL_TYPES, N_EDGE_TYPES

@pytest.fixture
def encoder_config():
    """Standard encoder configuration."""
    return {
        "d_input": 32,
        "d_hidden": 32,
        "d_output": 32,
        "n_heads": 2,
        "n_layers": 2,
        "n_node_types": 4,
        "n_edge_types": 2,
        "edge_dim": 1,
        "dropout": 0.0,
    }

@pytest.fixture
def encoder(encoder_config):
    from src.models.branches.hgt_encoder_tensor import HGTEncoderTensor
    return HGTEncoderTensor(**encoder_config)

@pytest.fixture
def sample_batch(encoder_config):
    """Sample batched input (flat edge format)."""
    B, N = 3, encoder_config["n_node_types"]
    d = encoder_config["d_input"]
    n_et = encoder_config["n_edge_types"]
    edges_per_sample = 8
    x = torch.randn(B, N, d)
    src_parts, dst_parts, type_parts = [], [], []
    for b in range(B):
        offset = b * N
        src_parts.append(torch.randint(0, N, (edges_per_sample,)) + offset)
        dst_parts.append(torch.randint(0, N, (edges_per_sample,)) + offset)
        type_parts.append(torch.randint(0, n_et, (edges_per_sample,)))
    edge_index = torch.stack([torch.cat(src_parts), torch.cat(dst_parts)])
    edge_type = torch.cat(type_parts)
    edge_attr = torch.rand(B * edges_per_sample, 1)
    return x, edge_index, edge_type, edge_attr

class TestHGTEncoderTensorInit:
    """Test initialization."""

    def test_creates_input_projection(self, encoder, encoder_config):
        N = encoder_config["n_node_types"]
        d_in = encoder_config["d_input"]
        d_hid = encoder_config["d_hidden"]
        assert encoder.input_proj_weight.shape == (N, d_in, d_hid)
        assert encoder.input_proj_bias.shape == (N, d_hid)

    def test_creates_layer_norms(self, encoder, encoder_config):
        n_layers = encoder_config["n_layers"]
        N = encoder_config["n_node_types"]
        d_hid = encoder_config["d_hidden"]
        assert encoder.ln_weight.shape == (n_layers, N, d_hid)
        assert encoder.ln_bias.shape == (n_layers, N, d_hid)

    def test_creates_layer_scales(self, encoder, encoder_config):
        n_layers = encoder_config["n_layers"]
        N = encoder_config["n_node_types"]
        assert len(encoder.layer_scales) == n_layers
        for scale in encoder.layer_scales:
            assert scale.shape == (N,)

    def test_creates_hgt_layers(self, encoder, encoder_config):
        assert len(encoder.hgt_layers) == encoder_config["n_layers"]

    def test_creates_output_projection_when_needed(self):
        from src.models.branches.hgt_encoder_tensor import HGTEncoderTensor
        enc = HGTEncoderTensor(d_input=32, d_hidden=32, d_output=64,
                               n_heads=2, n_layers=1, n_node_types=4,
                               n_edge_types=2, edge_dim=1)
        assert enc.output_proj_weight is not None
        assert enc.output_proj_weight.shape == (4, 32, 64)

    def test_no_output_projection_when_same(self, encoder):
        assert encoder.output_proj_weight is None

    def test_invalid_params(self):
        from src.models.branches.hgt_encoder_tensor import HGTEncoderTensor
        with pytest.raises(ValueError, match="d_input must be positive"):
            HGTEncoderTensor(0, 32, 32, 2, 1, 4, 2)
        with pytest.raises(ValueError, match="must be divisible by"):
            HGTEncoderTensor(32, 31, 32, 4, 1, 4, 2)

class TestHGTEncoderTensorForward:
    """Test forward pass."""

    def test_output_shape(self, encoder, sample_batch):
        x, edge_index, edge_type, edge_attr = sample_batch
        out = encoder(x, edge_index, edge_type, edge_attr)
        assert out.shape == x.shape  # [B, N, d_output]

    def test_output_no_nan(self, encoder, sample_batch):
        x, edge_index, edge_type, edge_attr = sample_batch
        out = encoder(x, edge_index, edge_type, edge_attr)
        assert torch.isfinite(out).all()

    def test_zero_edges(self, encoder_config):
        from src.models.branches.hgt_encoder_tensor import HGTEncoderTensor
        enc = HGTEncoderTensor(**encoder_config)
        B, N, d = 2, encoder_config["n_node_types"], encoder_config["d_input"]

        x = torch.randn(B, N, d)
        edge_index = torch.zeros(2, 0, dtype=torch.long)
        edge_type = torch.zeros(0, dtype=torch.long)
        edge_attr = torch.zeros(0, 1)

        out = enc(x, edge_index, edge_type, edge_attr)
        assert out.shape == (B, N, encoder_config["d_output"])
        assert torch.isfinite(out).all()

    def test_gradient_flow(self, encoder, sample_batch):
        x, edge_index, edge_type, edge_attr = sample_batch
        x.requires_grad_(True)
        out = encoder(x, edge_index, edge_type, edge_attr)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None
        for name, param in encoder.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_return_attention(self, encoder, sample_batch):
        x, edge_index, edge_type, edge_attr = sample_batch
        E_total = edge_index.shape[1]
        out, attn_list = encoder(x, edge_index, edge_type, edge_attr,
                                  return_attention=True)
        assert attn_list is not None
        assert len(attn_list) == encoder.n_layers
        H = encoder.n_heads
        for attn in attn_list:
            assert attn.shape == (E_total, H)

    def test_get_layer_scales(self, encoder, encoder_config):
        scales = encoder.get_layer_scales()
        assert 'scales' in scales
        assert scales['scales'].shape == (encoder_config["n_layers"],
                                          encoder_config["n_node_types"])

    def test_production_dimensions(self):
        from src.models.branches.hgt_encoder_tensor import HGTEncoderTensor
        enc = HGTEncoderTensor(
            d_input=128, d_hidden=128, d_output=128,
            n_heads=4, n_layers=3,
            n_node_types=N_CELL_TYPES, n_edge_types=N_EDGE_TYPES,
            edge_dim=1, dropout=0.0,
        )
        B, E_per = 4, 50
        x = torch.randn(B, N_CELL_TYPES, 128)
        src_parts, dst_parts, type_parts = [], [], []
        for b in range(B):
            offset = b * N_CELL_TYPES
            src_parts.append(torch.randint(0, N_CELL_TYPES, (E_per,)) + offset)
            dst_parts.append(torch.randint(0, N_CELL_TYPES, (E_per,)) + offset)
            type_parts.append(torch.randint(0, N_EDGE_TYPES, (E_per,)))
        edge_index = torch.stack([torch.cat(src_parts), torch.cat(dst_parts)])
        edge_type = torch.cat(type_parts)
        edge_attr = torch.rand(B * E_per, 1)

        out = enc(x, edge_index, edge_type, edge_attr)
        assert out.shape == (B, N_CELL_TYPES, 128)
        assert torch.isfinite(out).all()

    def test_gradient_checkpointing(self, encoder_config):
        from src.models.branches.hgt_encoder_tensor import HGTEncoderTensor
        enc = HGTEncoderTensor(**encoder_config, use_gradient_checkpointing=True)
        enc.train()

        B, N = 2, encoder_config["n_node_types"]
        d = encoder_config["d_input"]
        n_et = encoder_config["n_edge_types"]
        edges_per_sample = 5
        x = torch.randn(B, N, d, requires_grad=True)
        src_parts, dst_parts, type_parts = [], [], []
        for b in range(B):
            offset = b * N
            src_parts.append(torch.randint(0, N, (edges_per_sample,)) + offset)
            dst_parts.append(torch.randint(0, N, (edges_per_sample,)) + offset)
            type_parts.append(torch.randint(0, n_et, (edges_per_sample,)))
        edge_index = torch.stack([torch.cat(src_parts), torch.cat(dst_parts)])
        edge_type = torch.cat(type_parts)
        edge_attr = torch.rand(B * edges_per_sample, 1)

        out = enc(x, edge_index, edge_type, edge_attr)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None

class TestHGTEncoderTensorFlatEdges:
    """Test HGTEncoderTensor with flat (non-padded) edge format."""

    @pytest.fixture
    def flat_encoder_config(self):
        return {
            "d_input": 32,
            "d_hidden": 32,
            "d_output": 32,
            "n_heads": 2,
            "n_layers": 2,
            "n_node_types": 4,
            "n_edge_types": 2,
            "edge_dim": 1,
            "dropout": 0.0,
        }

    @pytest.fixture
    def flat_batch(self, flat_encoder_config):
        """Build flat edge tensors with batch-offset node indices for 3 samples."""
        B = 3
        N = flat_encoder_config["n_node_types"]
        d = flat_encoder_config["d_input"]
        n_et = flat_encoder_config["n_edge_types"]

        x = torch.randn(B, N, d)

        # Per-sample edge counts: 3, 5, 2 edges
        edges_per_sample = [3, 5, 2]
        src_list, dst_list, etype_list, eattr_list = [], [], [], []
        for b, n_edges in enumerate(edges_per_sample):
            offset = b * N
            src_list.append(torch.randint(0, N, (n_edges,)) + offset)
            dst_list.append(torch.randint(0, N, (n_edges,)) + offset)
            etype_list.append(torch.randint(0, n_et, (n_edges,)))
            eattr_list.append(torch.rand(n_edges, 1))

        edge_index = torch.stack([torch.cat(src_list), torch.cat(dst_list)])  # [2, E_total]
        edge_type = torch.cat(etype_list)  # [E_total]
        edge_attr = torch.cat(eattr_list)  # [E_total, 1]

        return x, edge_index, edge_type, edge_attr

    def test_flat_output_shape(self, flat_encoder_config, flat_batch):
        from src.models.branches.hgt_encoder_tensor import HGTEncoderTensor

        enc = HGTEncoderTensor(**flat_encoder_config)
        x, edge_index, edge_type, edge_attr = flat_batch
        B, N, d_out = x.shape[0], flat_encoder_config["n_node_types"], flat_encoder_config["d_output"]

        out = enc(x, edge_index, edge_type, edge_attr)  # edge_counts=None (flat)
        assert out.shape == (B, N, d_out)
        assert torch.isfinite(out).all()

    def test_flat_gradient_checkpointing(self, flat_encoder_config, flat_batch):
        from src.models.branches.hgt_encoder_tensor import HGTEncoderTensor

        enc = HGTEncoderTensor(**flat_encoder_config, use_gradient_checkpointing=True)
        enc.train()
        x, edge_index, edge_type, edge_attr = flat_batch
        x = x.clone().requires_grad_(True)

        out = enc(x, edge_index, edge_type, edge_attr)  # edge_counts=None (flat)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None
        for name, param in enc.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
