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
    """Sample batched input: B=3, N=4 node types, 6 edges per sample (flat)."""
    B, N = 3, conv_config["n_node_types"]
    d = conv_config["in_channels"]
    n_et = conv_config["n_edge_types"]
    edges_per_sample = 6
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
        x, edge_index, edge_type, edge_attr = sample_batch
        out = conv(x, edge_index, edge_type, edge_attr)
        assert out.shape == x.shape

    def test_output_no_nan(self, conv, sample_batch):
        x, edge_index, edge_type, edge_attr = sample_batch
        out = conv(x, edge_index, edge_type, edge_attr)
        assert torch.isfinite(out).all()

    def test_zero_edges_returns_zero(self, conv, conv_config):
        B, N, d = 2, conv_config["n_node_types"], conv_config["in_channels"]
        x = torch.randn(B, N, d)
        edge_index = torch.zeros(2, 0, dtype=torch.long)
        edge_type = torch.zeros(0, dtype=torch.long)
        edge_attr = torch.zeros(0, 1)
        out = conv(x, edge_index, edge_type, edge_attr)
        assert out.shape == (B, N, d)
        assert (out == 0).all()

    def test_gradient_flow(self, conv, sample_batch):
        x, edge_index, edge_type, edge_attr = sample_batch
        x.requires_grad_(True)
        out = conv(x, edge_index, edge_type, edge_attr)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        for name, param in conv.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_return_attention(self, conv, sample_batch):
        x, edge_index, edge_type, edge_attr = sample_batch
        E_total = edge_index.shape[1]
        H = conv.heads
        out, attn = conv(x, edge_index, edge_type, edge_attr, return_attention=True)
        assert attn.shape == (E_total, H)

    def test_attention_sums_to_one_per_dst(self, conv_config):
        from src.models.components.hgt_conv_tensor import HGTConvTensor
        conv = HGTConvTensor(**conv_config)
        B, N = 1, conv_config["n_node_types"]
        d = conv_config["in_channels"]
        x = torch.randn(B, N, d)
        edge_index = torch.tensor([[1, 2, 3], [0, 0, 0]])
        edge_type = torch.zeros(3, dtype=torch.long)
        edge_attr = torch.rand(3, 1)
        _, attn = conv(x, edge_index, edge_type, edge_attr, return_attention=True)
        attn_sum = attn[:, :].sum(dim=0)
        assert torch.allclose(attn_sum, torch.ones_like(attn_sum), atol=1e-5)

    def test_production_dimensions(self):
        from src.models.components.hgt_conv_tensor import HGTConvTensor
        conv = HGTConvTensor(
            in_channels=128, out_channels=128,
            n_node_types=N_CELL_TYPES, n_edge_types=N_EDGE_TYPES,
            heads=4, edge_dim=1, dropout=0.0,
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
        out = conv(x, edge_index, edge_type, edge_attr)
        assert out.shape == (B, N_CELL_TYPES, 128)
        assert torch.isfinite(out).all()

class TestHGTConvTensorFlatEdges:
    """Tests for the flat/concatenated edge format."""

    @pytest.fixture
    def flat_batch(self, conv_config):
        """Sample flat-edge input: B=3, varying edges per sample with batch offsets."""
        B = 3
        N = conv_config["n_node_types"]
        d = conv_config["in_channels"]
        n_et = conv_config["n_edge_types"]

        x = torch.randn(B, N, d)

        # Varying edges per sample: 5, 3, 4 = 12 total
        edges_per_sample = [5, 3, 4]
        src_list, dst_list, et_list, ea_list = [], [], [], []
        for b, n_edges in enumerate(edges_per_sample):
            offset = b * N
            src_list.append(torch.randint(0, N, (n_edges,)) + offset)
            dst_list.append(torch.randint(0, N, (n_edges,)) + offset)
            et_list.append(torch.randint(0, n_et, (n_edges,)))
            ea_list.append(torch.rand(n_edges, 1))

        edge_index = torch.stack([torch.cat(src_list), torch.cat(dst_list)])  # [2, 12]
        edge_type = torch.cat(et_list)  # [12]
        edge_attr = torch.cat(ea_list)  # [12, 1]

        return x, edge_index, edge_type, edge_attr

    def test_flat_output_shape(self, conv, conv_config, flat_batch):
        x, edge_index, edge_type, edge_attr = flat_batch
        B, N = x.shape[0], x.shape[1]
        out = conv(x, edge_index, edge_type, edge_attr)
        assert out.shape == (B, N, conv_config["out_channels"])

    def test_flat_output_no_nan(self, conv, flat_batch):
        x, edge_index, edge_type, edge_attr = flat_batch
        out = conv(x, edge_index, edge_type, edge_attr)
        assert torch.isfinite(out).all()

    def test_flat_zero_edges(self, conv, conv_config):
        B, N, d = 2, conv_config["n_node_types"], conv_config["in_channels"]
        x = torch.randn(B, N, d)
        edge_index = torch.zeros(2, 0, dtype=torch.long)
        edge_type = torch.zeros(0, dtype=torch.long)
        edge_attr = torch.zeros(0, 1)
        out = conv(x, edge_index, edge_type, edge_attr)
        assert out.shape == (B, N, conv_config["out_channels"])
        assert (out == 0).all()

    def test_flat_gradient_flow(self, conv, flat_batch):
        x, edge_index, edge_type, edge_attr = flat_batch
        x.requires_grad_(True)
        out = conv(x, edge_index, edge_type, edge_attr)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        for name, param in conv.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_flat_return_attention(self, conv, flat_batch):
        x, edge_index, edge_type, edge_attr = flat_batch
        E_total = edge_index.shape[1]
        H = conv.heads
        out, attn = conv(x, edge_index, edge_type, edge_attr, return_attention=True)
        assert attn.shape == (E_total, H)
