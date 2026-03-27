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
    """Sample batched input."""
    B, N, E = 3, encoder_config["n_node_types"], 8
    d = encoder_config["d_input"]
    n_et = encoder_config["n_edge_types"]

    x = torch.randn(B, N, d)
    edge_index = torch.randint(0, N, (B, 2, E))
    edge_type = torch.randint(0, n_et, (B, E))
    edge_attr = torch.rand(B, E, 1)
    edge_counts = torch.tensor([4, 8, 2])

    return x, edge_index, edge_type, edge_attr, edge_counts


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
        x, edge_index, edge_type, edge_attr, edge_counts = sample_batch
        out = encoder(x, edge_index, edge_type, edge_attr, edge_counts)
        assert out.shape == x.shape  # [B, N, d_output]

    def test_output_no_nan(self, encoder, sample_batch):
        x, edge_index, edge_type, edge_attr, edge_counts = sample_batch
        out = encoder(x, edge_index, edge_type, edge_attr, edge_counts)
        assert torch.isfinite(out).all()

    def test_zero_edges(self, encoder_config):
        from src.models.branches.hgt_encoder_tensor import HGTEncoderTensor
        enc = HGTEncoderTensor(**encoder_config)
        B, N, d = 2, encoder_config["n_node_types"], encoder_config["d_input"]

        x = torch.randn(B, N, d)
        edge_index = torch.zeros(B, 2, 0, dtype=torch.long)
        edge_type = torch.zeros(B, 0, dtype=torch.long)
        edge_attr = torch.zeros(B, 0, 1)
        edge_counts = torch.zeros(B, dtype=torch.long)

        out = enc(x, edge_index, edge_type, edge_attr, edge_counts)
        assert out.shape == (B, N, encoder_config["d_output"])
        assert torch.isfinite(out).all()

    def test_gradient_flow(self, encoder, sample_batch):
        x, edge_index, edge_type, edge_attr, edge_counts = sample_batch
        x.requires_grad_(True)
        out = encoder(x, edge_index, edge_type, edge_attr, edge_counts)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None
        for name, param in encoder.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_return_attention(self, encoder, sample_batch):
        x, edge_index, edge_type, edge_attr, edge_counts = sample_batch
        out, attn_list = encoder(x, edge_index, edge_type, edge_attr,
                                  edge_counts, return_attention=True)
        assert attn_list is not None
        assert len(attn_list) == encoder.n_layers
        B, E = x.shape[0], edge_index.shape[2]
        H = encoder.n_heads
        for attn in attn_list:
            assert attn.shape == (B, E, H)

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
        B, E = 4, 50
        x = torch.randn(B, N_CELL_TYPES, 128)
        edge_index = torch.randint(0, N_CELL_TYPES, (B, 2, E))
        edge_type = torch.randint(0, N_EDGE_TYPES, (B, E))
        edge_attr = torch.rand(B, E, 1)
        edge_counts = torch.randint(10, E + 1, (B,))

        out = enc(x, edge_index, edge_type, edge_attr, edge_counts)
        assert out.shape == (B, N_CELL_TYPES, 128)
        assert torch.isfinite(out).all()

    def test_gradient_checkpointing(self, encoder_config):
        from src.models.branches.hgt_encoder_tensor import HGTEncoderTensor
        enc = HGTEncoderTensor(**encoder_config, use_gradient_checkpointing=True)
        enc.train()

        B, N, E = 2, encoder_config["n_node_types"], 5
        d = encoder_config["d_input"]
        n_et = encoder_config["n_edge_types"]
        x = torch.randn(B, N, d, requires_grad=True)
        edge_index = torch.randint(0, N, (B, 2, E))
        edge_type = torch.randint(0, n_et, (B, E))
        edge_attr = torch.rand(B, E, 1)
        edge_counts = torch.tensor([3, 5])

        out = enc(x, edge_index, edge_type, edge_attr, edge_counts)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
