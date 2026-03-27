"""Unit tests for HGTConvTensor — batched tensor-native HGT convolution."""

import pytest
import torch

from src.data.constants import N_CELL_TYPES, N_EDGE_TYPES


@pytest.fixture
def conv_config():
    """Small conv configuration for testing."""
    return {
        "in_channels": 32,
        "out_channels": 32,
        "n_node_types": 4,
        "n_edge_types": 2,
        "heads": 2,
        "edge_dim": 1,
        "dropout": 0.0,
    }


@pytest.fixture
def conv(conv_config):
    from src.models.components.hgt_conv_tensor import HGTConvTensor
    return HGTConvTensor(**conv_config)


@pytest.fixture
def sample_batch(conv_config):
    """Sample batched input: B=3, N=4 node types, E=6 max edges."""
    B, N, E = 3, conv_config["n_node_types"], 6
    d = conv_config["in_channels"]
    n_et = conv_config["n_edge_types"]

    x = torch.randn(B, N, d)
    edge_index = torch.randint(0, N, (B, 2, E))
    edge_type = torch.randint(0, n_et, (B, E))
    edge_attr = torch.rand(B, E, 1)
    edge_counts = torch.tensor([4, 6, 2])

    return x, edge_index, edge_type, edge_attr, edge_counts


class TestHGTConvTensorInit:
    def test_creates_qkv_weights(self, conv, conv_config):
        N = conv_config["n_node_types"]
        d_in = conv_config["in_channels"]
        d_out = conv_config["out_channels"]
        assert conv.q_weight.shape == (N, d_in, d_out)
        assert conv.q_bias.shape == (N, d_out)
        assert conv.k_weight.shape == (N, d_in, d_out)
        assert conv.v_weight.shape == (N, d_in, d_out)

    def test_creates_relation_weights(self, conv, conv_config):
        n_et = conv_config["n_edge_types"]
        H = conv_config["heads"]
        dk = conv_config["out_channels"] // H
        assert conv.w_att.shape == (n_et, H, dk, dk)
        assert conv.w_msg.shape == (n_et, H, dk, dk)

    def test_creates_edge_projections(self, conv, conv_config):
        H = conv_config["heads"]
        assert conv.edge_lin.in_features == conv_config["edge_dim"]
        assert conv.edge_lin.out_features == H
        assert conv.edge_scale_lin.in_features == conv_config["edge_dim"]
        assert conv.edge_scale_lin.out_features == 1

    def test_creates_output_projection(self, conv, conv_config):
        d_out = conv_config["out_channels"]
        assert conv.out_lin.in_features == d_out
        assert conv.out_lin.out_features == d_out

    def test_invalid_params(self):
        from src.models.components.hgt_conv_tensor import HGTConvTensor
        with pytest.raises(ValueError, match="in_channels must be positive"):
            HGTConvTensor(0, 32, 4, 2)
        with pytest.raises(ValueError, match="must be divisible by"):
            HGTConvTensor(32, 31, 4, 2, heads=4)

    def test_no_edge_dim(self):
        from src.models.components.hgt_conv_tensor import HGTConvTensor
        conv = HGTConvTensor(32, 32, 4, 2, edge_dim=None)
        assert conv.edge_lin is None
        assert conv.edge_scale_lin is None


class TestHGTConvTensorForward:
    def test_output_shape(self, conv, sample_batch):
        x, edge_index, edge_type, edge_attr, edge_counts = sample_batch
        out = conv(x, edge_index, edge_type, edge_attr, edge_counts)
        assert out.shape == x.shape

    def test_output_no_nan(self, conv, sample_batch):
        x, edge_index, edge_type, edge_attr, edge_counts = sample_batch
        out = conv(x, edge_index, edge_type, edge_attr, edge_counts)
        assert torch.isfinite(out).all()

    def test_zero_edges_returns_zero(self, conv, conv_config):
        B, N, d = 2, conv_config["n_node_types"], conv_config["in_channels"]
        x = torch.randn(B, N, d)
        edge_index = torch.zeros(B, 2, 0, dtype=torch.long)
        edge_type = torch.zeros(B, 0, dtype=torch.long)
        edge_attr = torch.zeros(B, 0, 1)
        edge_counts = torch.zeros(B, dtype=torch.long)
        out = conv(x, edge_index, edge_type, edge_attr, edge_counts)
        assert out.shape == (B, N, d)
        assert (out == 0).all()

    def test_padding_edges_ignored(self, conv, conv_config):
        B, N, d = 1, conv_config["n_node_types"], conv_config["in_channels"]
        n_et = conv_config["n_edge_types"]
        torch.manual_seed(42)
        x = torch.randn(B, N, d)
        E = 8
        edge_index = torch.randint(0, N, (B, 2, E))
        edge_type = torch.randint(0, n_et, (B, E))
        edge_attr = torch.rand(B, E, 1)
        edge_counts = torch.tensor([3])
        out_with_padding = conv(x, edge_index, edge_type, edge_attr, edge_counts)
        edge_index_trim = edge_index[:, :, :3]
        edge_type_trim = edge_type[:, :3]
        edge_attr_trim = edge_attr[:, :3, :]
        edge_counts_same = torch.tensor([3])
        out_no_padding = conv(x, edge_index_trim, edge_type_trim, edge_attr_trim, edge_counts_same)
        assert torch.allclose(out_with_padding, out_no_padding, atol=1e-6)

    def test_gradient_flow(self, conv, sample_batch):
        x, edge_index, edge_type, edge_attr, edge_counts = sample_batch
        x.requires_grad_(True)
        out = conv(x, edge_index, edge_type, edge_attr, edge_counts)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        for name, param in conv.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_return_attention(self, conv, sample_batch):
        x, edge_index, edge_type, edge_attr, edge_counts = sample_batch
        out, attn = conv(x, edge_index, edge_type, edge_attr, edge_counts, return_attention=True)
        B, E = edge_index.shape[0], edge_index.shape[2]
        H = conv.heads
        assert attn.shape == (B, E, H)

    def test_attention_sums_to_one_per_dst(self, conv_config):
        from src.models.components.hgt_conv_tensor import HGTConvTensor
        conv = HGTConvTensor(**conv_config)
        B, N = 1, conv_config["n_node_types"]
        d = conv_config["in_channels"]
        x = torch.randn(B, N, d)
        edge_index = torch.tensor([[[1, 2, 3], [0, 0, 0]]])
        edge_type = torch.zeros(B, 3, dtype=torch.long)
        edge_attr = torch.rand(B, 3, 1)
        edge_counts = torch.tensor([3])
        _, attn = conv(x, edge_index, edge_type, edge_attr, edge_counts, return_attention=True)
        attn_sum = attn[0, :, :].sum(dim=0)
        assert torch.allclose(attn_sum, torch.ones_like(attn_sum), atol=1e-5)

    def test_production_dimensions(self):
        from src.models.components.hgt_conv_tensor import HGTConvTensor
        conv = HGTConvTensor(
            in_channels=128, out_channels=128,
            n_node_types=N_CELL_TYPES, n_edge_types=N_EDGE_TYPES,
            heads=4, edge_dim=1, dropout=0.0,
        )
        B, E = 4, 50
        x = torch.randn(B, N_CELL_TYPES, 128)
        edge_index = torch.randint(0, N_CELL_TYPES, (B, 2, E))
        edge_type = torch.randint(0, N_EDGE_TYPES, (B, E))
        edge_attr = torch.rand(B, E, 1)
        edge_counts = torch.randint(10, E + 1, (B,))
        out = conv(x, edge_index, edge_type, edge_attr, edge_counts)
        assert out.shape == (B, N_CELL_TYPES, 128)
        assert torch.isfinite(out).all()
