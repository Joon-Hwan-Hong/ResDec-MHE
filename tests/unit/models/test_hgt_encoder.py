"""
Unit tests for HGTEncoder with true heterogeneous node types.

Tests cover:
- Basic functionality with dict-based inputs
- True heterogeneous graph with 31 cell types
- Edge attribute (LIANA magnitude) support
- Attention weight extraction
- Gradient flow
- Batched processing
- Edge cases and error handling
"""

import pytest
import torch

from src.data.constants import ALL_EDGE_TYPES, CELL_TYPE_ORDER, N_CELL_TYPES
from src.models.branches.hgt_encoder import HGTEncoder, HGTEncoderBatched
from src.models.components.hgt_conv import HGTConvWithEdgeAttr


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def encoder_config():
    """Standard encoder configuration."""
    return {
        "d_input": 128,
        "d_hidden": 64,
        "d_output": 64,
        "n_heads": 4,
        "n_layers": 2,
        "dropout": 0.1,
        "edge_dim": 1,
    }


@pytest.fixture
def small_encoder_config():
    """Small encoder for faster tests."""
    return {
        "d_input": 32,
        "d_hidden": 16,
        "d_output": 16,
        "n_heads": 2,
        "n_layers": 1,
        "dropout": 0.0,
        "edge_dim": 1,
    }


@pytest.fixture
def mini_node_types():
    """Small set of node types for faster tests."""
    return ["Astrocyte", "Oligodendrocyte", "Microglia", "CGE interneuron"]


@pytest.fixture
def mini_edge_categories():
    """Small set of edge categories for faster tests."""
    return ["Secreted_Signaling", "ECM_Receptor"]


@pytest.fixture
def encoder(encoder_config):
    """Standard HGTEncoder instance."""
    return HGTEncoder(**encoder_config)


@pytest.fixture
def small_encoder(small_encoder_config, mini_node_types, mini_edge_categories):
    """Small HGTEncoder for faster tests."""
    return HGTEncoder(
        **small_encoder_config,
        node_types=mini_node_types,
        edge_categories=mini_edge_categories,
    )


@pytest.fixture
def sample_graph_dict(small_encoder_config, mini_node_types, mini_edge_categories):
    """Sample heterogeneous graph data using dict format."""
    d_input = small_encoder_config["d_input"]
    edge_dim = small_encoder_config["edge_dim"]

    # Create x_dict: one embedding per cell type
    x_dict = {
        node_type: torch.randn(1, d_input)
        for node_type in mini_node_types
    }

    # Create edge_index_dict and edge_attr_dict
    edge_index_dict = {}
    edge_attr_dict = {}

    # Add some edges between node types
    for i, src in enumerate(mini_node_types):
        for j, dst in enumerate(mini_node_types):
            if i != j:  # Skip self-loops for variety
                for edge_cat in mini_edge_categories:
                    edge_type = (src, edge_cat, dst)
                    # Random number of edges (1-3)
                    n_edges = torch.randint(1, 4, (1,)).item()
                    # Since each node type has 1 node, indices are 0
                    edge_index_dict[edge_type] = torch.zeros(2, n_edges, dtype=torch.long)
                    edge_attr_dict[edge_type] = torch.rand(n_edges, edge_dim)

    return x_dict, edge_index_dict, edge_attr_dict


@pytest.fixture
def batched_encoder(small_encoder_config, mini_node_types, mini_edge_categories):
    """Batched HGTEncoder instance."""
    return HGTEncoderBatched(
        **small_encoder_config,
        node_types=mini_node_types,
        edge_categories=mini_edge_categories,
    )


# ============================================================================
# HGTConvWithEdgeAttr Tests
# ============================================================================


class TestHGTConvWithEdgeAttr:
    """Test the custom HGT convolution layer."""

    def test_initialization(self, mini_node_types, mini_edge_categories):
        """Test layer initializes correctly."""
        conv = HGTConvWithEdgeAttr(
            in_channels=32,
            out_channels=32,
            node_types=mini_node_types,
            edge_categories=mini_edge_categories,
            heads=2,
            edge_dim=1,
        )
        assert conv.in_channels == 32
        assert conv.out_channels == 32
        assert conv.heads == 2
        assert conv.edge_dim == 1
        assert len(conv.node_types) == len(mini_node_types)

    def test_forward_shape(self, mini_node_types, mini_edge_categories):
        """Test forward pass produces correct output shapes."""
        conv = HGTConvWithEdgeAttr(
            in_channels=32,
            out_channels=32,
            node_types=mini_node_types,
            edge_categories=mini_edge_categories,
            heads=2,
            edge_dim=1,
        )

        # Create simple input
        x_dict = {nt: torch.randn(1, 32) for nt in mini_node_types}
        edge_index_dict = {
            (mini_node_types[0], mini_edge_categories[0], mini_node_types[1]):
                torch.tensor([[0], [0]])
        }
        edge_attr_dict = {
            (mini_node_types[0], mini_edge_categories[0], mini_node_types[1]):
                torch.rand(1, 1)
        }

        out_dict, attn = conv(x_dict, edge_index_dict, edge_attr_dict)

        for nt in mini_node_types:
            assert nt in out_dict
            assert out_dict[nt].shape == (1, 32)

    def test_attention_extraction(self, mini_node_types, mini_edge_categories):
        """Test attention weight extraction."""
        conv = HGTConvWithEdgeAttr(
            in_channels=32,
            out_channels=32,
            node_types=mini_node_types,
            edge_categories=mini_edge_categories,
            heads=2,
            edge_dim=1,
        )

        x_dict = {nt: torch.randn(1, 32) for nt in mini_node_types}
        edge_type = (mini_node_types[0], mini_edge_categories[0], mini_node_types[1])
        edge_index_dict = {edge_type: torch.tensor([[0], [0]])}
        edge_attr_dict = {edge_type: torch.rand(1, 1)}

        out_dict, attn_dict = conv(
            x_dict, edge_index_dict, edge_attr_dict, return_attention=True
        )

        assert attn_dict is not None
        assert edge_type in attn_dict
        # Attention should be [n_edges, n_heads]
        assert attn_dict[edge_type].shape == (1, 2)

    def test_invalid_parameters(self, mini_node_types, mini_edge_categories):
        """Test error on invalid parameters."""
        with pytest.raises(ValueError, match="in_channels must be positive"):
            HGTConvWithEdgeAttr(
                in_channels=0,
                out_channels=32,
                node_types=mini_node_types,
                edge_categories=mini_edge_categories,
            )

        with pytest.raises(ValueError, match="must be divisible by"):
            HGTConvWithEdgeAttr(
                in_channels=32,
                out_channels=31,  # Not divisible by heads=4
                node_types=mini_node_types,
                edge_categories=mini_edge_categories,
                heads=4,
            )


# ============================================================================
# Basic Functionality Tests
# ============================================================================


class TestBasicFunctionality:
    """Test basic encoder operations."""

    def test_initialization(self, encoder_config):
        """Test encoder initializes correctly."""
        encoder = HGTEncoder(**encoder_config)
        assert encoder.d_input == encoder_config["d_input"]
        assert encoder.d_hidden == encoder_config["d_hidden"]
        assert encoder.d_output == encoder_config["d_output"]
        assert encoder.n_heads == encoder_config["n_heads"]
        assert encoder.n_layers == encoder_config["n_layers"]
        assert encoder.edge_dim == encoder_config["edge_dim"]

    def test_default_node_types(self, encoder_config):
        """Test default node types are 31 cell types."""
        encoder = HGTEncoder(**encoder_config)
        assert encoder.node_types == list(CELL_TYPE_ORDER)
        assert encoder.n_node_types == N_CELL_TYPES

    def test_default_edge_categories(self, encoder_config):
        """Test default edge categories are CellChatDB types."""
        encoder = HGTEncoder(**encoder_config)
        assert encoder.edge_categories == ALL_EDGE_TYPES
        assert encoder.n_edge_types == len(ALL_EDGE_TYPES)

    def test_custom_node_types(self, small_encoder_config, mini_node_types):
        """Test custom node types."""
        encoder = HGTEncoder(**small_encoder_config, node_types=mini_node_types)
        assert encoder.node_types == mini_node_types
        assert encoder.n_node_types == len(mini_node_types)

    def test_custom_edge_categories(self, small_encoder_config, mini_edge_categories):
        """Test custom edge categories."""
        encoder = HGTEncoder(**small_encoder_config, edge_categories=mini_edge_categories)
        assert encoder.edge_categories == mini_edge_categories
        assert encoder.n_edge_types == len(mini_edge_categories)

    def test_forward_shape(self, small_encoder, sample_graph_dict, mini_node_types):
        """Test forward pass produces correct output shapes."""
        x_dict, edge_index_dict, edge_attr_dict = sample_graph_dict

        output_dict, _ = small_encoder(x_dict, edge_index_dict, edge_attr_dict)

        for node_type in mini_node_types:
            assert node_type in output_dict
            assert output_dict[node_type].shape == (1, small_encoder.d_output)

    def test_forward_without_edge_attrs(self, small_encoder, sample_graph_dict, mini_node_types):
        """Test forward pass without edge attributes."""
        x_dict, edge_index_dict, _ = sample_graph_dict

        # Should work without edge_attr_dict
        output_dict, _ = small_encoder(x_dict, edge_index_dict, edge_attr_dict=None)

        for node_type in mini_node_types:
            assert node_type in output_dict

    def test_output_projection(self, small_encoder_config, mini_node_types, mini_edge_categories):
        """Test output projection when d_output != d_hidden."""
        config = {**small_encoder_config, "d_output": 32}  # Different from d_hidden=16
        encoder = HGTEncoder(
            **config,
            node_types=mini_node_types,
            edge_categories=mini_edge_categories,
        )

        x_dict = {nt: torch.randn(1, config["d_input"]) for nt in mini_node_types}
        edge_index_dict = {}  # Empty graph

        output_dict, _ = encoder(x_dict, edge_index_dict)

        for node_type in mini_node_types:
            assert output_dict[node_type].shape == (1, 32)


# ============================================================================
# Attention Weight Tests
# ============================================================================


class TestAttentionWeights:
    """Test attention weight extraction."""

    def test_return_attention(self, small_encoder, sample_graph_dict):
        """Test attention weights are returned when requested."""
        x_dict, edge_index_dict, edge_attr_dict = sample_graph_dict

        output_dict, attn_weights = small_encoder(
            x_dict, edge_index_dict, edge_attr_dict, return_attention=True
        )

        assert attn_weights is not None
        assert isinstance(attn_weights, list)
        assert len(attn_weights) == small_encoder.n_layers

    def test_attention_not_returned_by_default(self, small_encoder, sample_graph_dict):
        """Test attention weights not returned by default."""
        x_dict, edge_index_dict, edge_attr_dict = sample_graph_dict

        output_dict, attn_weights = small_encoder(x_dict, edge_index_dict, edge_attr_dict)

        assert attn_weights is None


# ============================================================================
# Edge Attribute Tests
# ============================================================================


class TestEdgeAttributes:
    """Test edge attribute (LIANA magnitude) handling."""

    def test_edge_attr_affects_output(self, small_encoder_config, mini_node_types, mini_edge_categories):
        """Test that edge attributes affect the output.

        Note: With only one edge per edge type to a target, softmax normalizes
        to 1.0. We need multiple source nodes of the SAME type sending to the
        same target for edge_attr to affect attention weights via softmax.
        """
        # Use more source nodes to have meaningful attention competition
        encoder = HGTEncoder(
            **small_encoder_config,
            node_types=mini_node_types,
            edge_categories=mini_edge_categories,
        )
        encoder.eval()

        torch.manual_seed(42)
        # Create 2 source nodes for Astrocyte, 1 target node for Oligodendrocyte
        x_dict = {
            mini_node_types[0]: torch.randn(2, small_encoder_config["d_input"]),  # 2 Astrocytes
            mini_node_types[1]: torch.randn(1, small_encoder_config["d_input"]),  # 1 Oligodendrocyte
            mini_node_types[2]: torch.randn(1, small_encoder_config["d_input"]),
            mini_node_types[3]: torch.randn(1, small_encoder_config["d_input"]),
        }

        # Two edges from different Astrocyte nodes to the same Oligodendrocyte
        edge_type = (mini_node_types[0], mini_edge_categories[0], mini_node_types[1])
        edge_index_dict = {
            edge_type: torch.tensor([[0, 1], [0, 0]]),  # Astrocyte 0,1 -> Oligodendrocyte 0
        }

        # Low weight on first edge, high on second
        edge_attr_dict_low = {edge_type: torch.tensor([[0.1], [10.0]])}
        output_low, _ = encoder(x_dict, edge_index_dict, edge_attr_dict_low)

        # High weight on first edge, low on second (swapped)
        edge_attr_dict_high = {edge_type: torch.tensor([[10.0], [0.1]])}
        output_high, _ = encoder(x_dict, edge_index_dict, edge_attr_dict_high)

        # Outputs should differ due to different attention distributions
        dst_type = mini_node_types[1]
        assert not torch.allclose(output_low[dst_type], output_high[dst_type], atol=1e-6)

    def test_edge_attr_gradient_flow(self, small_encoder_config, mini_node_types, mini_edge_categories):
        """Test gradients flow through edge attributes."""
        encoder = HGTEncoder(
            **small_encoder_config,
            node_types=mini_node_types,
            edge_categories=mini_edge_categories,
        )

        x_dict = {nt: torch.randn(1, small_encoder_config["d_input"]) for nt in mini_node_types}

        edge_type = (mini_node_types[0], mini_edge_categories[0], mini_node_types[1])
        edge_index_dict = {edge_type: torch.tensor([[0], [0]])}
        edge_attr = torch.rand(1, 1, requires_grad=True)
        edge_attr_dict = {edge_type: edge_attr}

        output_dict, _ = encoder(x_dict, edge_index_dict, edge_attr_dict)

        # Sum all outputs and backprop
        loss = sum(out.sum() for out in output_dict.values())
        loss.backward()

        # Edge attr should have gradients
        assert edge_attr.grad is not None


# ============================================================================
# Gradient Flow Tests
# ============================================================================


class TestGradientFlow:
    """Test gradient flow through the encoder."""

    def test_gradients_flow_to_input(self, small_encoder, sample_graph_dict, mini_node_types):
        """Test gradients flow back to input."""
        x_dict, edge_index_dict, edge_attr_dict = sample_graph_dict

        # Make inputs require grad
        for nt in x_dict:
            x_dict[nt].requires_grad = True

        output_dict, _ = small_encoder(x_dict, edge_index_dict, edge_attr_dict)
        loss = sum(out.sum() for out in output_dict.values())
        loss.backward()

        for nt in mini_node_types:
            assert x_dict[nt].grad is not None

    def test_gradients_to_hgt_layers(self, small_encoder, sample_graph_dict):
        """Test gradients reach HGT layer parameters."""
        x_dict, edge_index_dict, edge_attr_dict = sample_graph_dict

        output_dict, _ = small_encoder(x_dict, edge_index_dict, edge_attr_dict)
        loss = sum(out.sum() for out in output_dict.values())
        loss.backward()

        # Check first HGT layer has gradients
        for param in small_encoder.hgt_layers[0].parameters():
            if param.requires_grad:
                assert param.grad is not None


# ============================================================================
# Batched Encoder Tests
# ============================================================================


class TestBatchedEncoder:
    """Test batched HGT encoder."""

    def test_batched_initialization(self, batched_encoder, small_encoder_config):
        """Test batched encoder initializes correctly."""
        assert batched_encoder.encoder.d_input == small_encoder_config["d_input"]

    def test_batched_forward_shape(
        self, batched_encoder, small_encoder_config, mini_node_types, mini_edge_categories
    ):
        """Test batched forward pass produces correct shape."""
        batch_size = 3
        d_input = small_encoder_config["d_input"]

        # Create batch of graphs
        x_dict_list = [
            {nt: torch.randn(1, d_input) for nt in mini_node_types}
            for _ in range(batch_size)
        ]

        edge_index_dict_list = [
            {
                (mini_node_types[0], mini_edge_categories[0], mini_node_types[1]):
                    torch.tensor([[0], [0]])
            }
            for _ in range(batch_size)
        ]

        edge_attr_dict_list = [
            {
                (mini_node_types[0], mini_edge_categories[0], mini_node_types[1]):
                    torch.rand(1, 1)
            }
            for _ in range(batch_size)
        ]

        output_dict, _ = batched_encoder(
            x_dict_list, edge_index_dict_list, edge_attr_dict_list
        )

        for node_type in mini_node_types:
            assert node_type in output_dict
            # Shape: (batch, n_nodes_per_type, d_output)
            assert output_dict[node_type].shape == (batch_size, 1, small_encoder_config["d_output"])

    def test_batched_gradient_flow(
        self, batched_encoder, small_encoder_config, mini_node_types, mini_edge_categories
    ):
        """Test gradient flow in batched encoder."""
        batch_size = 2
        d_input = small_encoder_config["d_input"]

        x_dict_list = [
            {nt: torch.randn(1, d_input, requires_grad=True) for nt in mini_node_types}
            for _ in range(batch_size)
        ]

        edge_index_dict_list = [
            {
                (mini_node_types[0], mini_edge_categories[0], mini_node_types[1]):
                    torch.tensor([[0], [0]])
            }
            for _ in range(batch_size)
        ]

        output_dict, _ = batched_encoder(x_dict_list, edge_index_dict_list)
        loss = sum(out.sum() for out in output_dict.values())
        loss.backward()

        for x_dict in x_dict_list:
            for nt in mini_node_types:
                assert x_dict[nt].grad is not None

    def test_batched_mismatched_list_length(
        self, batched_encoder, small_encoder_config, mini_node_types, mini_edge_categories
    ):
        """Test error on mismatched list lengths."""
        d_input = small_encoder_config["d_input"]

        x_dict_list = [
            {nt: torch.randn(1, d_input) for nt in mini_node_types}
            for _ in range(4)
        ]
        edge_index_dict_list = [{} for _ in range(3)]  # Mismatched length

        with pytest.raises(ValueError, match="must match batch size"):
            batched_encoder(x_dict_list, edge_index_dict_list)

    def test_edge_attr_dict_list_length_mismatch(
        self, batched_encoder, small_encoder_config, mini_node_types, mini_edge_categories
    ):
        """Mismatched edge_index and edge_attr list lengths should raise ValueError."""
        d_input = small_encoder_config["d_input"]

        # Create 2 x_dict and edge_index entries
        x_dict_list = [
            {nt: torch.randn(1, d_input) for nt in mini_node_types}
            for _ in range(2)
        ]
        edge_index_dict_list = [{} for _ in range(2)]

        # But 3 edge_attr entries (mismatch)
        edge_attr_dict_list = [{} for _ in range(3)]

        with pytest.raises(ValueError, match="must match batch size"):
            batched_encoder(x_dict_list, edge_index_dict_list, edge_attr_dict_list)

    def test_batched_mismatched_node_types(
        self, batched_encoder, small_encoder_config, mini_node_types
    ):
        """Test warning and zero-fill when samples have different node types."""
        d_input = small_encoder_config["d_input"]

        # Sample 0: has all node types
        # Sample 1: missing one node type
        x_dict_list = [
            {nt: torch.randn(1, d_input) for nt in mini_node_types},
            {nt: torch.randn(1, d_input) for nt in mini_node_types[:-1]},  # Missing last type
        ]
        edge_index_dict_list = [{}, {}]

        # Missing node types are logged at DEBUG level and zero-filled
        output_dict, _ = batched_encoder(x_dict_list, edge_index_dict_list)
        # Missing node type should be zero-filled in sample 1
        missing_type = mini_node_types[-1]
        assert missing_type in output_dict
        # Sample 1's missing type should have zero input (zero-filled)
        # After encoder processing, it may not be exactly zero due to bias terms,
        # but the output should exist and have the right shape
        assert output_dict[missing_type].shape[0] == 2  # batch size

    def test_batched_return_attention(
        self, batched_encoder, small_encoder_config, mini_node_types, mini_edge_categories
    ):
        """Test that return_attention=True returns attention weights per sample."""
        batch_size = 3
        d_input = small_encoder_config["d_input"]

        # Create batch of graphs with edges so attention is non-trivial
        x_dict_list = [
            {nt: torch.randn(1, d_input) for nt in mini_node_types}
            for _ in range(batch_size)
        ]

        edge_type = (mini_node_types[0], mini_edge_categories[0], mini_node_types[1])
        edge_index_dict_list = [
            {edge_type: torch.tensor([[0], [0]])}
            for _ in range(batch_size)
        ]

        edge_attr_dict_list = [
            {edge_type: torch.rand(1, 1)}
            for _ in range(batch_size)
        ]

        output_dict, all_attention = batched_encoder(
            x_dict_list, edge_index_dict_list, edge_attr_dict_list,
            return_attention=True,
        )

        # all_attention should be a list with one entry per sample
        assert all_attention is not None
        assert isinstance(all_attention, list)
        assert len(all_attention) == batch_size

        # Each sample's attention should be a list of layer attention dicts
        n_layers = small_encoder_config["n_layers"]
        for sample_attn in all_attention:
            assert isinstance(sample_attn, list)
            assert len(sample_attn) == n_layers

            # Each layer's attention is a dict mapping edge_type -> weights
            for layer_attn in sample_attn:
                assert isinstance(layer_attn, dict)
                assert edge_type in layer_attn
                # Attention weights shape: [n_edges, n_heads]
                attn_weights = layer_attn[edge_type]
                assert attn_weights.shape == (1, small_encoder_config["n_heads"])
                # Attention values should be valid probabilities
                assert (attn_weights >= 0).all()
                assert (attn_weights <= 1 + 1e-5).all()

    def test_batched_return_attention_default_none(
        self, batched_encoder, small_encoder_config, mini_node_types, mini_edge_categories
    ):
        """Test that attention is None when return_attention is not set."""
        batch_size = 2
        d_input = small_encoder_config["d_input"]

        x_dict_list = [
            {nt: torch.randn(1, d_input) for nt in mini_node_types}
            for _ in range(batch_size)
        ]
        edge_index_dict_list = [{} for _ in range(batch_size)]

        output_dict, all_attention = batched_encoder(
            x_dict_list, edge_index_dict_list,
        )

        assert all_attention is None


# ============================================================================
# Batched Proxy Property Tests (HGT-A3)
# ============================================================================


class TestBatchedProxyProperties:
    """Test that HGTEncoderBatched proxy properties correctly delegate to inner encoder.

    Covers HGT-A3: the 4 proxy @property decorators and the get_edge_type_index
    method had zero test coverage.
    """

    def test_n_edge_types_delegates(
        self, batched_encoder, mini_edge_categories
    ):
        """n_edge_types property should return same value as inner encoder."""
        assert batched_encoder.n_edge_types == batched_encoder.encoder.n_edge_types
        assert batched_encoder.n_edge_types == len(mini_edge_categories)

    def test_n_node_types_delegates(
        self, batched_encoder, mini_node_types
    ):
        """n_node_types property should return same value as inner encoder."""
        assert batched_encoder.n_node_types == batched_encoder.encoder.n_node_types
        assert batched_encoder.n_node_types == len(mini_node_types)

    def test_node_types_delegates(
        self, batched_encoder, mini_node_types
    ):
        """node_types property should return same list as inner encoder."""
        result = batched_encoder.node_types
        assert result == batched_encoder.encoder.node_types
        assert result == mini_node_types
        assert isinstance(result, list)
        assert all(isinstance(nt, str) for nt in result)

    def test_edge_categories_delegates(
        self, batched_encoder, mini_edge_categories
    ):
        """edge_categories property should return same list as inner encoder."""
        result = batched_encoder.edge_categories
        assert result == batched_encoder.encoder.edge_categories
        assert result == mini_edge_categories
        assert isinstance(result, list)
        assert all(isinstance(ec, str) for ec in result)

    def test_get_edge_type_index_delegates(
        self, batched_encoder, mini_edge_categories
    ):
        """get_edge_type_index() should return same index as inner encoder."""
        for i, cat in enumerate(mini_edge_categories):
            batched_result = batched_encoder.get_edge_type_index(cat)
            inner_result = batched_encoder.encoder.get_edge_type_index(cat)
            assert batched_result == inner_result
            assert batched_result == i

    def test_get_edge_type_index_invalid_category(self, batched_encoder):
        """get_edge_type_index() with invalid category should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown edge category"):
            batched_encoder.get_edge_type_index("Nonexistent_Category")

    def test_proxy_properties_with_default_types(self, small_encoder_config):
        """Proxy properties should work when using default node/edge types."""
        batched = HGTEncoderBatched(**small_encoder_config)

        # Should delegate to inner encoder which uses defaults
        assert batched.n_node_types == batched.encoder.n_node_types
        assert batched.n_edge_types == batched.encoder.n_edge_types
        assert batched.node_types == batched.encoder.node_types
        assert batched.edge_categories == batched.encoder.edge_categories

        # Verify types are correct
        assert batched.n_node_types == len(CELL_TYPE_ORDER)
        assert batched.n_edge_types == len(ALL_EDGE_TYPES)
        assert batched.node_types == list(CELL_TYPE_ORDER)
        assert batched.edge_categories == ALL_EDGE_TYPES


# ============================================================================
# Edge Cases and Error Handling Tests
# ============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_invalid_d_input(self):
        """Test error on invalid d_input."""
        with pytest.raises(ValueError, match="d_input must be positive"):
            HGTEncoder(d_input=0, d_hidden=64, d_output=64)

    def test_invalid_d_hidden(self):
        """Test error on invalid d_hidden."""
        with pytest.raises(ValueError, match="d_hidden must be positive"):
            HGTEncoder(d_input=64, d_hidden=0, d_output=64)

    def test_invalid_d_output(self):
        """Test error on invalid d_output."""
        with pytest.raises(ValueError, match="d_output must be positive"):
            HGTEncoder(d_input=64, d_hidden=64, d_output=0)

    def test_invalid_n_heads(self):
        """Test error on invalid n_heads."""
        with pytest.raises(ValueError, match="n_heads must be positive"):
            HGTEncoder(d_input=64, d_hidden=64, d_output=64, n_heads=0)

    def test_invalid_n_layers(self):
        """Test error on invalid n_layers."""
        with pytest.raises(ValueError, match="n_layers must be positive"):
            HGTEncoder(d_input=64, d_hidden=64, d_output=64, n_layers=0)

    def test_d_hidden_not_divisible_by_heads(self):
        """Test error when d_hidden not divisible by n_heads."""
        with pytest.raises(ValueError, match="must be divisible by"):
            HGTEncoder(d_input=64, d_hidden=63, d_output=64, n_heads=4)

    def test_empty_graph(self, small_encoder, mini_node_types, small_encoder_config):
        """Test with empty graph (no edges)."""
        x_dict = {nt: torch.randn(1, small_encoder_config["d_input"]) for nt in mini_node_types}
        edge_index_dict = {}  # No edges

        output_dict, _ = small_encoder(x_dict, edge_index_dict)

        for node_type in mini_node_types:
            assert node_type in output_dict
            assert output_dict[node_type].shape == (1, small_encoder.d_output)

    def test_partial_node_types(self, small_encoder, mini_node_types, small_encoder_config):
        """Test with only some node types present in input."""
        # Only provide 2 of 4 node types
        x_dict = {
            mini_node_types[0]: torch.randn(1, small_encoder_config["d_input"]),
            mini_node_types[1]: torch.randn(1, small_encoder_config["d_input"]),
        }
        edge_index_dict = {}

        output_dict, _ = small_encoder(x_dict, edge_index_dict)

        # Should only have outputs for provided node types
        assert mini_node_types[0] in output_dict
        assert mini_node_types[1] in output_dict

    def test_get_edge_type_index(self, small_encoder, mini_edge_categories):
        """Test edge type index lookup."""
        for i, cat in enumerate(mini_edge_categories):
            assert small_encoder.get_edge_type_index(cat) == i

    def test_invalid_edge_category(self, small_encoder):
        """Test error on invalid edge category."""
        with pytest.raises(ValueError, match="Unknown edge category"):
            small_encoder.get_edge_type_index("Invalid_Category")


# ============================================================================
# Numerical Stability Tests
# ============================================================================


class TestNumericalStability:
    """Test numerical stability."""

    def test_no_nan_output(self, small_encoder, sample_graph_dict):
        """Test no NaN in output."""
        x_dict, edge_index_dict, edge_attr_dict = sample_graph_dict
        output_dict, _ = small_encoder(x_dict, edge_index_dict, edge_attr_dict)

        for node_type, output in output_dict.items():
            assert not torch.isnan(output).any(), f"NaN in output for {node_type}"

    def test_no_inf_output(self, small_encoder, sample_graph_dict):
        """Test no Inf in output."""
        x_dict, edge_index_dict, edge_attr_dict = sample_graph_dict
        output_dict, _ = small_encoder(x_dict, edge_index_dict, edge_attr_dict)

        for node_type, output in output_dict.items():
            assert not torch.isinf(output).any(), f"Inf in output for {node_type}"

    def test_large_input_values(self, small_encoder, mini_node_types, small_encoder_config):
        """Test stability with large input values."""
        x_dict = {
            nt: torch.randn(1, small_encoder_config["d_input"]) * 100
            for nt in mini_node_types
        }
        edge_index_dict = {}

        output_dict, _ = small_encoder(x_dict, edge_index_dict)

        for output in output_dict.values():
            assert not torch.isnan(output).any()
            assert not torch.isinf(output).any()

    def test_small_input_values(self, small_encoder, mini_node_types, small_encoder_config):
        """Test stability with small input values."""
        x_dict = {
            nt: torch.randn(1, small_encoder_config["d_input"]) * 1e-6
            for nt in mini_node_types
        }
        edge_index_dict = {}

        output_dict, _ = small_encoder(x_dict, edge_index_dict)

        for output in output_dict.values():
            assert not torch.isnan(output).any()


# ============================================================================
# Determinism Tests
# ============================================================================


class TestDeterminism:
    """Test deterministic behavior."""

    def test_eval_mode_determinism(self, small_encoder, sample_graph_dict):
        """Test deterministic output in eval mode."""
        small_encoder.eval()
        x_dict, edge_index_dict, edge_attr_dict = sample_graph_dict

        output1, _ = small_encoder(x_dict, edge_index_dict, edge_attr_dict)
        output2, _ = small_encoder(x_dict, edge_index_dict, edge_attr_dict)

        for node_type in output1:
            assert torch.allclose(output1[node_type], output2[node_type])


# ============================================================================
# Extra Repr Test
# ============================================================================


class TestExtraRepr:
    """Test string representation."""

    def test_extra_repr(self, small_encoder, small_encoder_config):
        """Test extra_repr contains key info."""
        repr_str = small_encoder.extra_repr()
        assert str(small_encoder_config["d_input"]) in repr_str
        assert str(small_encoder_config["d_hidden"]) in repr_str
        assert str(small_encoder_config["d_output"]) in repr_str
        assert "n_node_types" in repr_str
        assert "edge_dim" in repr_str


# ============================================================================
# Integration Tests with Full Cell Types
# ============================================================================


class TestFullCellTypes:
    """Test with full 31 cell types (slower but important)."""

    def test_full_31_cell_types(self, encoder_config):
        """Test encoder works with all 31 cell types."""
        encoder = HGTEncoder(**encoder_config)

        assert encoder.n_node_types == N_CELL_TYPES

        # Create input with all 31 cell types
        x_dict = {
            ct: torch.randn(1, encoder_config["d_input"])
            for ct in CELL_TYPE_ORDER
        }

        # Add a few edges
        edge_index_dict = {
            ("Astrocyte", "Secreted_Signaling", "Microglia"): torch.tensor([[0], [0]]),
            ("Oligodendrocyte", "ECM_Receptor", "Astrocyte"): torch.tensor([[0], [0]]),
        }
        edge_attr_dict = {
            ("Astrocyte", "Secreted_Signaling", "Microglia"): torch.rand(1, 1),
            ("Oligodendrocyte", "ECM_Receptor", "Astrocyte"): torch.rand(1, 1),
        }

        output_dict, attn = encoder(
            x_dict, edge_index_dict, edge_attr_dict, return_attention=True
        )

        # All 31 cell types should have outputs
        assert len(output_dict) == N_CELL_TYPES
        for ct in CELL_TYPE_ORDER:
            assert ct in output_dict
            assert output_dict[ct].shape == (1, encoder_config["d_output"])

        # Attention should be returned
        assert attn is not None
        assert len(attn) == encoder_config["n_layers"]


# ============================================================================
# Negative Tests (Invalid Inputs)
# ============================================================================


class TestNegativeInputs:
    """Test rejection of invalid inputs."""

    def test_empty_node_types_raises(self, small_encoder_config):
        """Empty node_types list should raise ValueError."""
        with pytest.raises(ValueError, match="node_types must not be empty"):
            HGTConvWithEdgeAttr(
                in_channels=32,
                out_channels=32,
                node_types=[],
                edge_categories=["Type_A"],
            )

    def test_empty_edge_categories_raises(self, small_encoder_config):
        """Empty edge_categories list should raise ValueError."""
        with pytest.raises(ValueError, match="edge_categories must not be empty"):
            HGTConvWithEdgeAttr(
                in_channels=32,
                out_channels=32,
                node_types=["Type_A"],
                edge_categories=[],
            )

    def test_edge_index_out_of_bounds_raises(
        self, small_encoder, mini_node_types, small_encoder_config
    ):
        """Edge indices exceeding node count should raise IndexError.

        PyTorch indexing (e.g., q[dst_idx]) with out-of-bounds indices raises
        IndexError. This is the correct behavior — silent corruption would be worse.
        """
        x_dict = {nt: torch.randn(1, small_encoder_config["d_input"]) for nt in mini_node_types}

        # Edge index points to node 5, but only 1 node exists per type
        edge_type = (mini_node_types[0], "Secreted_Signaling", mini_node_types[1])
        edge_index_dict = {edge_type: torch.tensor([[5], [0]])}  # src=5 out of bounds

        with pytest.raises((IndexError, RuntimeError)):
            small_encoder(x_dict, edge_index_dict)

    def test_mismatched_edge_attr_dimension(
        self, mini_node_types, mini_edge_categories
    ):
        """edge_attr with wrong feature dimension should cause issues."""
        encoder = HGTEncoder(
            d_input=32,
            d_hidden=16,
            d_output=16,
            n_heads=2,
            n_layers=1,
            dropout=0.0,
            edge_dim=1,  # Expects 1-dimensional edge features
            node_types=mini_node_types,
            edge_categories=mini_edge_categories,
        )

        x_dict = {nt: torch.randn(1, 32) for nt in mini_node_types}  # d_input=32
        edge_type = (mini_node_types[0], mini_edge_categories[0], mini_node_types[1])
        edge_index_dict = {edge_type: torch.tensor([[0], [0]])}

        # Wrong edge_attr dimension (3 instead of 1)
        edge_attr_dict = {edge_type: torch.rand(1, 3)}

        # Should raise due to dimension mismatch in linear layer
        with pytest.raises(RuntimeError):
            encoder(x_dict, edge_index_dict, edge_attr_dict)

    def test_edge_attr_missing_for_some_edge_types_works(
        self, small_encoder, mini_node_types, mini_edge_categories, small_encoder_config
    ):
        """Partial edge_attr_dict should work (missing types get no bias)."""
        x_dict = {nt: torch.randn(1, small_encoder_config["d_input"]) for nt in mini_node_types}

        edge_type_1 = (mini_node_types[0], mini_edge_categories[0], mini_node_types[1])
        edge_type_2 = (mini_node_types[1], mini_edge_categories[1], mini_node_types[2])

        edge_index_dict = {
            edge_type_1: torch.tensor([[0], [0]]),
            edge_type_2: torch.tensor([[0], [0]]),
        }

        # Only provide edge_attr for one edge type
        edge_attr_dict = {edge_type_1: torch.rand(1, 1)}

        # Should work without error
        output_dict, _ = small_encoder(x_dict, edge_index_dict, edge_attr_dict)
        assert len(output_dict) > 0

    def test_wrong_input_dtype_float64_raises(
        self, small_encoder, mini_node_types, small_encoder_config
    ):
        """Float64 input should raise RuntimeError (dtype mismatch with float32 weights)."""
        x_dict = {
            nt: torch.randn(1, small_encoder_config["d_input"], dtype=torch.float64)
            for nt in mini_node_types
        }
        edge_index_dict = {}

        # Should raise due to dtype mismatch (model weights are float32)
        with pytest.raises(RuntimeError, match="must have the same dtype"):
            small_encoder(x_dict, edge_index_dict)


# ============================================================================
# Extended Edge Cases
# ============================================================================


class TestEdgeCasesExtended:
    """Extended edge case coverage."""

    def test_self_loops(self, small_encoder, mini_node_types, small_encoder_config):
        """Self-loops (node to itself) should be handled correctly."""
        x_dict = {nt: torch.randn(1, small_encoder_config["d_input"]) for nt in mini_node_types}

        # Self-loop: Astrocyte node 0 to itself
        edge_type = (mini_node_types[0], "Secreted_Signaling", mini_node_types[0])
        edge_index_dict = {edge_type: torch.tensor([[0], [0]])}
        edge_attr_dict = {edge_type: torch.rand(1, 1)}

        output_dict, _ = small_encoder(x_dict, edge_index_dict, edge_attr_dict)

        # Should not crash and produce valid output
        assert mini_node_types[0] in output_dict
        assert not torch.isnan(output_dict[mini_node_types[0]]).any()

    def test_all_edges_to_single_node(
        self, small_encoder_config, mini_node_types, mini_edge_categories
    ):
        """Many edges converging on one node (stress test aggregation)."""
        encoder = HGTEncoder(
            **small_encoder_config,
            node_types=mini_node_types,
            edge_categories=mini_edge_categories,
        )

        # 10 source nodes of type 0, 1 target node of type 1
        x_dict = {
            mini_node_types[0]: torch.randn(10, small_encoder_config["d_input"]),
            mini_node_types[1]: torch.randn(1, small_encoder_config["d_input"]),
            mini_node_types[2]: torch.randn(1, small_encoder_config["d_input"]),
            mini_node_types[3]: torch.randn(1, small_encoder_config["d_input"]),
        }

        # All 10 source nodes connect to the single target
        edge_type = (mini_node_types[0], mini_edge_categories[0], mini_node_types[1])
        edge_index_dict = {
            edge_type: torch.tensor([
                list(range(10)),  # src: 0,1,2,...,9
                [0] * 10,         # dst: all to node 0
            ])
        }
        edge_attr_dict = {edge_type: torch.rand(10, 1)}

        output_dict, attn = encoder(
            x_dict, edge_index_dict, edge_attr_dict, return_attention=True
        )

        assert output_dict[mini_node_types[1]].shape == (1, small_encoder_config["d_output"])
        assert not torch.isnan(output_dict[mini_node_types[1]]).any()

    def test_disconnected_graph_preserves_features(
        self, small_encoder, mini_node_types, small_encoder_config
    ):
        """Nodes with no edges should retain their features via residual."""
        small_encoder.eval()

        x_dict = {nt: torch.randn(1, small_encoder_config["d_input"]) for nt in mini_node_types}
        edge_index_dict = {}  # No edges

        output_dict, _ = small_encoder(x_dict, edge_index_dict)

        # Output should be non-zero (residual preserves projected input)
        for nt in mini_node_types:
            assert output_dict[nt].abs().sum() > 0

    def test_bidirectional_edges(
        self, small_encoder, mini_node_types, mini_edge_categories, small_encoder_config
    ):
        """A→B and B→A both present should work correctly."""
        x_dict = {nt: torch.randn(1, small_encoder_config["d_input"]) for nt in mini_node_types}

        # Bidirectional edges
        edge_type_ab = (mini_node_types[0], mini_edge_categories[0], mini_node_types[1])
        edge_type_ba = (mini_node_types[1], mini_edge_categories[0], mini_node_types[0])

        edge_index_dict = {
            edge_type_ab: torch.tensor([[0], [0]]),
            edge_type_ba: torch.tensor([[0], [0]]),
        }
        edge_attr_dict = {
            edge_type_ab: torch.rand(1, 1),
            edge_type_ba: torch.rand(1, 1),
        }

        output_dict, _ = small_encoder(x_dict, edge_index_dict, edge_attr_dict)

        # Both nodes should have valid outputs
        assert not torch.isnan(output_dict[mini_node_types[0]]).any()
        assert not torch.isnan(output_dict[mini_node_types[1]]).any()

    def test_multiple_nodes_per_type(
        self, small_encoder_config, mini_node_types, mini_edge_categories
    ):
        """Multiple nodes of each type (not just 1)."""
        encoder = HGTEncoder(
            **small_encoder_config,
            node_types=mini_node_types,
            edge_categories=mini_edge_categories,
        )

        # 5 nodes per type
        x_dict = {
            nt: torch.randn(5, small_encoder_config["d_input"])
            for nt in mini_node_types
        }

        # Various edges between nodes
        edge_type = (mini_node_types[0], mini_edge_categories[0], mini_node_types[1])
        edge_index_dict = {
            edge_type: torch.tensor([
                [0, 1, 2, 3, 4],  # src nodes 0-4 of type 0
                [0, 1, 2, 3, 4],  # dst nodes 0-4 of type 1
            ])
        }
        edge_attr_dict = {edge_type: torch.rand(5, 1)}

        output_dict, _ = encoder(x_dict, edge_index_dict, edge_attr_dict)

        # All types should have 5 output nodes
        for nt in mini_node_types:
            assert output_dict[nt].shape == (5, small_encoder_config["d_output"])

    def test_asymmetric_node_counts(
        self, small_encoder_config, mini_node_types, mini_edge_categories
    ):
        """Different node counts per type."""
        encoder = HGTEncoder(
            **small_encoder_config,
            node_types=mini_node_types,
            edge_categories=mini_edge_categories,
        )

        # Different counts: 10, 3, 1, 7
        x_dict = {
            mini_node_types[0]: torch.randn(10, small_encoder_config["d_input"]),
            mini_node_types[1]: torch.randn(3, small_encoder_config["d_input"]),
            mini_node_types[2]: torch.randn(1, small_encoder_config["d_input"]),
            mini_node_types[3]: torch.randn(7, small_encoder_config["d_input"]),
        }

        edge_index_dict = {}  # No edges for simplicity

        output_dict, _ = encoder(x_dict, edge_index_dict)

        assert output_dict[mini_node_types[0]].shape[0] == 10
        assert output_dict[mini_node_types[1]].shape[0] == 3
        assert output_dict[mini_node_types[2]].shape[0] == 1
        assert output_dict[mini_node_types[3]].shape[0] == 7


# ============================================================================
# Mechanistic Correctness Tests
# ============================================================================


class TestMechanisticCorrectness:
    """Verify implementation matches scientific intentions.

    These tests ensure the HGT mechanism (attention, type-specificity,
    edge features) works as intended for cell-cell communication modeling.
    """

    # === Cell Type Specificity ===

    def test_different_cell_types_have_different_projections(
        self, small_encoder, mini_node_types
    ):
        """Q/K/V projections should be distinct per cell type.

        Each cell type should have its own learned representation space.
        """
        conv = small_encoder.hgt_layers[0]

        # Get two different cell type keys
        key_0 = conv._node_type_to_key[mini_node_types[0]]
        key_1 = conv._node_type_to_key[mini_node_types[1]]

        # Their projection weights should be different (not shared)
        q_weight_0 = conv.q_lin[key_0].weight
        q_weight_1 = conv.q_lin[key_1].weight

        # Weights should not be identical (they're independently initialized)
        assert not torch.allclose(q_weight_0, q_weight_1)

    def test_cell_type_projection_independence(
        self, small_encoder, mini_node_types, small_encoder_config
    ):
        """Changing one cell type's input should not affect
        unconnected cell types' outputs.
        """
        small_encoder.eval()

        x_dict_1 = {nt: torch.randn(1, small_encoder_config["d_input"]) for nt in mini_node_types}

        # Change only Astrocyte's input
        x_dict_2 = {nt: x_dict_1[nt].clone() for nt in mini_node_types}
        x_dict_2[mini_node_types[0]] = torch.randn(1, small_encoder_config["d_input"])

        edge_index_dict = {}  # No edges - completely disconnected

        output_1, _ = small_encoder(x_dict_1, edge_index_dict)
        output_2, _ = small_encoder(x_dict_2, edge_index_dict)

        # Astrocyte output should change
        assert not torch.allclose(
            output_1[mini_node_types[0]],
            output_2[mini_node_types[0]]
        )

        # Other cell types should be unchanged (no edges connecting them)
        for nt in mini_node_types[1:]:
            assert torch.allclose(output_1[nt], output_2[nt])

    # === Edge Category Specificity ===

    def test_edge_categories_have_different_parameters(
        self, small_encoder, mini_edge_categories
    ):
        """Each edge category should have its own W_ATT and W_MSG."""
        conv = small_encoder.hgt_layers[0]

        key_0 = conv._edge_cat_to_key[mini_edge_categories[0]]
        key_1 = conv._edge_cat_to_key[mini_edge_categories[1]]

        # W_ATT should be different per category
        assert not torch.allclose(conv.w_att[key_0], conv.w_att[key_1])

        # W_MSG should be different per category
        assert not torch.allclose(conv.w_msg[key_0], conv.w_msg[key_1])

    def test_edge_categories_produce_different_outputs(
        self, small_encoder_config, mini_node_types, mini_edge_categories
    ):
        """Same edge endpoints with different categories should yield
        different outputs.
        """
        encoder = HGTEncoder(
            **small_encoder_config,
            node_types=mini_node_types,
            edge_categories=mini_edge_categories,
        )
        encoder.eval()

        torch.manual_seed(42)
        x_dict = {nt: torch.randn(1, small_encoder_config["d_input"]) for nt in mini_node_types}

        # Same edge but different categories
        edge_type_cat0 = (mini_node_types[0], mini_edge_categories[0], mini_node_types[1])
        edge_type_cat1 = (mini_node_types[0], mini_edge_categories[1], mini_node_types[1])

        edge_index_dict_cat0 = {edge_type_cat0: torch.tensor([[0], [0]])}
        edge_index_dict_cat1 = {edge_type_cat1: torch.tensor([[0], [0]])}

        edge_attr = torch.tensor([[1.0]])

        output_cat0, _ = encoder(x_dict, edge_index_dict_cat0, {edge_type_cat0: edge_attr})
        output_cat1, _ = encoder(x_dict, edge_index_dict_cat1, {edge_type_cat1: edge_attr})

        # Destination node should have different outputs
        assert not torch.allclose(
            output_cat0[mini_node_types[1]],
            output_cat1[mini_node_types[1]],
            atol=1e-5
        )

    # === LIANA Magnitude Effects ===

    def test_higher_liana_increases_attention_weight(
        self, small_encoder_config, mini_node_types, mini_edge_categories
    ):
        """Higher edge_attr value should increase attention weight.

        Scientific intent: stronger LIANA score = more confident
        interaction = higher attention.
        """
        encoder = HGTEncoder(
            **small_encoder_config,
            node_types=mini_node_types,
            edge_categories=mini_edge_categories,
        )
        encoder.eval()

        # Two source nodes competing for attention to one target
        x_dict = {
            mini_node_types[0]: torch.randn(2, small_encoder_config["d_input"]),
            mini_node_types[1]: torch.randn(1, small_encoder_config["d_input"]),
            mini_node_types[2]: torch.randn(1, small_encoder_config["d_input"]),
            mini_node_types[3]: torch.randn(1, small_encoder_config["d_input"]),
        }

        edge_type = (mini_node_types[0], mini_edge_categories[0], mini_node_types[1])
        edge_index_dict = {
            edge_type: torch.tensor([[0, 1], [0, 0]])  # Both src nodes -> same dst
        }

        # First edge has HIGH weight, second has LOW
        edge_attr_high_low = torch.tensor([[10.0], [0.1]])
        _, attn_high_low = encoder(
            x_dict, edge_index_dict, {edge_type: edge_attr_high_low}, return_attention=True
        )

        # First edge has LOW weight, second has HIGH
        edge_attr_low_high = torch.tensor([[0.1], [10.0]])
        _, attn_low_high = encoder(
            x_dict, edge_index_dict, {edge_type: edge_attr_low_high}, return_attention=True
        )

        # Get attention weights from first layer
        attn_weights_hl = attn_high_low[0][edge_type]  # [2, n_heads]
        attn_weights_lh = attn_low_high[0][edge_type]

        # Average across heads
        avg_attn_hl = attn_weights_hl.mean(dim=1)  # [2]
        avg_attn_lh = attn_weights_lh.mean(dim=1)

        # When edge 0 has high weight, it should have higher attention
        assert avg_attn_hl[0] > avg_attn_hl[1], "High LIANA edge should have higher attention"

        # When edge 1 has high weight, it should have higher attention
        assert avg_attn_lh[1] > avg_attn_lh[0], "High LIANA edge should have higher attention"

    def test_liana_magnitude_monotonic_effect(
        self, small_encoder_config, mini_node_types, mini_edge_categories
    ):
        """Attention should increase monotonically with edge_attr (all else equal)."""
        encoder = HGTEncoder(
            **small_encoder_config,
            node_types=mini_node_types,
            edge_categories=mini_edge_categories,
        )
        encoder.eval()

        x_dict = {
            mini_node_types[0]: torch.randn(3, small_encoder_config["d_input"]),
            mini_node_types[1]: torch.randn(1, small_encoder_config["d_input"]),
            mini_node_types[2]: torch.randn(1, small_encoder_config["d_input"]),
            mini_node_types[3]: torch.randn(1, small_encoder_config["d_input"]),
        }

        edge_type = (mini_node_types[0], mini_edge_categories[0], mini_node_types[1])
        edge_index_dict = {
            edge_type: torch.tensor([[0, 1, 2], [0, 0, 0]])
        }

        # Monotonically increasing LIANA scores
        edge_attr = torch.tensor([[1.0], [5.0], [10.0]])
        _, attn = encoder(
            x_dict, edge_index_dict, {edge_type: edge_attr}, return_attention=True
        )

        attn_weights = attn[0][edge_type].mean(dim=1)  # [3] averaged across heads

        # Attention should be monotonically increasing
        assert attn_weights[1] > attn_weights[0], "Attention should increase with LIANA"
        assert attn_weights[2] > attn_weights[1], "Attention should increase with LIANA"

    # === Message Passing Correctness ===

    def test_information_flows_only_along_edges(
        self, small_encoder_config, mini_node_types, mini_edge_categories
    ):
        """Perturbing an unconnected node should not change target's output."""
        encoder = HGTEncoder(
            **small_encoder_config,
            node_types=mini_node_types,
            edge_categories=mini_edge_categories,
        )
        encoder.eval()

        torch.manual_seed(42)
        x_dict_1 = {nt: torch.randn(1, small_encoder_config["d_input"]) for nt in mini_node_types}

        # Only edge: type[0] -> type[1]
        edge_type = (mini_node_types[0], mini_edge_categories[0], mini_node_types[1])
        edge_index_dict = {edge_type: torch.tensor([[0], [0]])}
        edge_attr_dict = {edge_type: torch.tensor([[1.0]])}

        output_1, _ = encoder(x_dict_1, edge_index_dict, edge_attr_dict)

        # Perturb type[2] which is NOT connected to type[1]
        x_dict_2 = {nt: x_dict_1[nt].clone() for nt in mini_node_types}
        x_dict_2[mini_node_types[2]] = torch.randn(1, small_encoder_config["d_input"])

        output_2, _ = encoder(x_dict_2, edge_index_dict, edge_attr_dict)

        # type[1]'s output should be unchanged (only receives from type[0])
        assert torch.allclose(
            output_1[mini_node_types[1]],
            output_2[mini_node_types[1]]
        )

    def test_no_edges_output_equals_projected_input(
        self, small_encoder, mini_node_types, small_encoder_config
    ):
        """With no incoming edges, output should come from residual path."""
        small_encoder.eval()

        x_dict = {nt: torch.randn(1, small_encoder_config["d_input"]) for nt in mini_node_types}
        edge_index_dict = {}  # No edges

        output_dict, _ = small_encoder(x_dict, edge_index_dict)

        # Each output should be the projected+normed input (residual only)
        # The output should be non-zero and finite
        for nt in mini_node_types:
            assert output_dict[nt].abs().sum() > 0
            assert not torch.isnan(output_dict[nt]).any()
            assert not torch.isinf(output_dict[nt]).any()

    def test_isolated_nodes_get_zero_conv_output(
        self, mini_node_types, mini_edge_categories
    ):
        """HGT conv should output zero for isolated nodes (no incoming edges).

        This tests the conv layer directly (not the encoder), verifying that
        'no LIANA edges = no communication contribution'. The encoder adds
        residual connections, so isolated nodes still get non-zero final output,
        but the conv layer's contribution should be exactly zero.
        """
        # Create conv layer directly
        d_model = 16
        conv = HGTConvWithEdgeAttr(
            in_channels=d_model,
            out_channels=d_model,
            heads=2,
            node_types=mini_node_types,
            edge_categories=mini_edge_categories,
            edge_dim=1,
        )
        conv.eval()

        # Input features for all node types
        x_dict = {nt: torch.randn(1, d_model) for nt in mini_node_types}

        # Only one edge: type[0] -> type[1]
        # type[2] is isolated (no incoming edges)
        edge_type = (mini_node_types[0], mini_edge_categories[0], mini_node_types[1])
        edge_index_dict = {edge_type: torch.tensor([[0], [0]])}
        edge_attr_dict = {edge_type: torch.tensor([[0.8]])}

        out_dict, _ = conv(x_dict, edge_index_dict, edge_attr_dict)

        # type[1] received a message, should be non-zero
        assert out_dict[mini_node_types[1]].abs().sum() > 0, (
            "Node receiving message should have non-zero conv output"
        )

        # type[2] is isolated, should be exactly zero (no communication)
        assert out_dict[mini_node_types[2]].abs().sum() == 0, (
            "Isolated node should have zero conv output (no LIANA edges = no communication)"
        )

        # type[0] is source but receives no incoming edges, should be zero
        assert out_dict[mini_node_types[0]].abs().sum() == 0, (
            "Source-only node should have zero conv output (no incoming edges)"
        )

    # === Attention Properties ===

    def test_attention_weights_sum_to_one_per_target(
        self, small_encoder_config, mini_node_types, mini_edge_categories
    ):
        """For each target node, attention over incoming edges should sum to 1."""
        encoder = HGTEncoder(
            **small_encoder_config,
            node_types=mini_node_types,
            edge_categories=mini_edge_categories,
        )
        encoder.eval()

        x_dict = {
            mini_node_types[0]: torch.randn(3, small_encoder_config["d_input"]),
            mini_node_types[1]: torch.randn(1, small_encoder_config["d_input"]),
            mini_node_types[2]: torch.randn(1, small_encoder_config["d_input"]),
            mini_node_types[3]: torch.randn(1, small_encoder_config["d_input"]),
        }

        edge_type = (mini_node_types[0], mini_edge_categories[0], mini_node_types[1])
        edge_index_dict = {
            edge_type: torch.tensor([[0, 1, 2], [0, 0, 0]])  # 3 edges to same target
        }
        edge_attr_dict = {edge_type: torch.rand(3, 1)}

        _, attn = encoder(
            x_dict, edge_index_dict, edge_attr_dict, return_attention=True
        )

        # Attention weights for this edge type
        attn_weights = attn[0][edge_type]  # [3, n_heads]

        # Sum across edges (dim=0) should equal 1 for each head
        attn_sum = attn_weights.sum(dim=0)
        assert torch.allclose(attn_sum, torch.ones_like(attn_sum), atol=1e-5)

    def test_attention_weights_non_negative(
        self, small_encoder, sample_graph_dict
    ):
        """All attention weights should be >= 0."""
        small_encoder.eval()
        x_dict, edge_index_dict, edge_attr_dict = sample_graph_dict

        _, attn = small_encoder(
            x_dict, edge_index_dict, edge_attr_dict, return_attention=True
        )

        for layer_attn in attn:
            for edge_type, weights in layer_attn.items():
                assert (weights >= 0).all(), f"Negative attention in {edge_type}"

    def test_attention_weights_bounded_by_one(
        self, small_encoder, sample_graph_dict
    ):
        """All attention weights should be <= 1."""
        small_encoder.eval()
        x_dict, edge_index_dict, edge_attr_dict = sample_graph_dict

        _, attn = small_encoder(
            x_dict, edge_index_dict, edge_attr_dict, return_attention=True
        )

        for layer_attn in attn:
            for edge_type, weights in layer_attn.items():
                assert (weights <= 1 + 1e-5).all(), f"Attention > 1 in {edge_type}"

    # === Heterogeneous Graph Properties ===

    def test_31_cell_types_all_have_parameters(self, encoder_config):
        """All 31 Allen ABC cell types should have learned projections."""
        encoder = HGTEncoder(**encoder_config)
        conv = encoder.hgt_layers[0]

        for cell_type in CELL_TYPE_ORDER:
            key = conv._node_type_to_key[cell_type]
            assert key in conv.q_lin, f"Missing Q projection for {cell_type}"
            assert key in conv.k_lin, f"Missing K projection for {cell_type}"
            assert key in conv.v_lin, f"Missing V projection for {cell_type}"

    def test_5_edge_categories_all_have_parameters(self, encoder_config):
        """All 5 CellChatDB categories should have W_ATT, W_MSG."""
        encoder = HGTEncoder(**encoder_config)
        conv = encoder.hgt_layers[0]

        for edge_cat in ALL_EDGE_TYPES:
            key = conv._edge_cat_to_key[edge_cat]
            assert key in conv.w_att, f"Missing W_ATT for {edge_cat}"
            assert key in conv.w_msg, f"Missing W_MSG for {edge_cat}"

    def test_cross_type_communication_uses_correct_projections(
        self, small_encoder_config, mini_node_types, mini_edge_categories
    ):
        """Edge A→B should use A's K,V and B's Q projections.

        This is the core HGT mechanism: queries come from target,
        keys/values come from source.
        """
        encoder = HGTEncoder(
            **small_encoder_config,
            node_types=mini_node_types,
            edge_categories=mini_edge_categories,
        )

        # Manually verify the forward pass uses correct projections
        # by checking that gradients flow to the expected parameters
        x_dict = {
            nt: torch.randn(1, small_encoder_config["d_input"], requires_grad=True)
            for nt in mini_node_types
        }

        src_type = mini_node_types[0]  # Astrocyte
        dst_type = mini_node_types[1]  # Oligodendrocyte

        edge_type = (src_type, mini_edge_categories[0], dst_type)
        edge_index_dict = {edge_type: torch.tensor([[0], [0]])}
        edge_attr_dict = {edge_type: torch.rand(1, 1)}

        output_dict, _ = encoder(x_dict, edge_index_dict, edge_attr_dict)

        # Only backprop from dst output
        # NOTE: Using MSE loss instead of plain sum() because LayerNorm has
        # a mathematical property where d(sum(output))/d(input) = 0 when
        # outputs are normalized to have mean 0. MSE breaks this symmetry
        # and matches what we use in actual training.
        target = torch.zeros_like(output_dict[dst_type])
        loss = torch.nn.functional.mse_loss(output_dict[dst_type], target)
        loss.backward()

        # Source node should have gradients (its K,V were used)
        assert x_dict[src_type].grad is not None
        assert x_dict[src_type].grad.abs().sum() > 0

        # Destination node should have gradients (its Q was used + residual)
        assert x_dict[dst_type].grad is not None
        assert x_dict[dst_type].grad.abs().sum() > 0


# ============================================================================
# Training Behavior Tests
# ============================================================================


class TestTrainingBehavior:
    """Verify training-related properties."""

    def test_all_parameters_receive_gradients(
        self, small_encoder, sample_graph_dict
    ):
        """Every learnable parameter should receive gradients during backprop."""
        x_dict, edge_index_dict, edge_attr_dict = sample_graph_dict

        # Make inputs require grad
        for nt in x_dict:
            x_dict[nt] = x_dict[nt].clone().requires_grad_(True)

        output_dict, _ = small_encoder(x_dict, edge_index_dict, edge_attr_dict)
        loss = sum(out.sum() for out in output_dict.values())
        loss.backward()

        # Check all parameters have gradients
        for name, param in small_encoder.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                # Gradient should be non-zero for most parameters
                # (some might be zero due to specific graph structure)

    def test_edge_attr_gradient_magnitude_non_trivial(
        self, mini_node_types, mini_edge_categories
    ):
        """Gradients w.r.t. edge_attr should be non-trivial.

        Note: The gradient through the full encoder can be small due to
        residual connections diluting the signal. We test the HGT conv
        layer directly for a stronger signal.
        """
        from src.models.components.hgt_conv import HGTConvWithEdgeAttr

        conv = HGTConvWithEdgeAttr(
            in_channels=32,
            out_channels=32,
            node_types=mini_node_types,
            edge_categories=mini_edge_categories,
            heads=2,
            edge_dim=1,
        )

        x_dict = {nt: torch.randn(2, 32) for nt in mini_node_types}

        edge_type = (mini_node_types[0], mini_edge_categories[0], mini_node_types[1])
        edge_index_dict = {edge_type: torch.tensor([[0, 1], [0, 0]])}
        edge_attr = torch.rand(2, 1, requires_grad=True)
        edge_attr_dict = {edge_type: edge_attr}

        out_dict, _ = conv(x_dict, edge_index_dict, edge_attr_dict)

        # Use only the target node's output for stronger gradient signal
        loss = out_dict[mini_node_types[1]].sum()
        loss.backward()

        assert edge_attr.grad is not None
        assert edge_attr.grad.abs().sum() > 1e-6, "Edge attr gradients are trivially small"

    def test_dropout_affects_training_mode(
        self, small_encoder_config, mini_node_types, mini_edge_categories
    ):
        """Training mode should produce different outputs across runs due to dropout."""
        encoder = HGTEncoder(
            **{**small_encoder_config, "dropout": 0.5},  # High dropout
            node_types=mini_node_types,
            edge_categories=mini_edge_categories,
        )
        encoder.train()  # Training mode

        torch.manual_seed(42)
        x_dict = {nt: torch.randn(1, small_encoder_config["d_input"]) for nt in mini_node_types}
        edge_index_dict = {}

        output_1, _ = encoder(x_dict, edge_index_dict)
        output_2, _ = encoder(x_dict, edge_index_dict)

        # Outputs should differ due to dropout
        any_different = any(
            not torch.allclose(output_1[nt], output_2[nt])
            for nt in mini_node_types
        )
        assert any_different, "Dropout should cause different outputs in training mode"

    def test_no_gradient_explosion(self, small_encoder, sample_graph_dict):
        """Gradient norms should stay bounded."""
        x_dict, edge_index_dict, edge_attr_dict = sample_graph_dict

        output_dict, _ = small_encoder(x_dict, edge_index_dict, edge_attr_dict)
        loss = sum(out.sum() for out in output_dict.values())
        loss.backward()

        # Check gradient norms are reasonable
        for name, param in small_encoder.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm()
                assert grad_norm < 1000, f"Gradient explosion in {name}: norm={grad_norm}"


# ============================================================================
# Reproducibility Tests
# ============================================================================


class TestReproducibility:
    """Verify reproducible behavior."""

    def test_same_seed_same_initialization(self, small_encoder_config, mini_node_types, mini_edge_categories):
        """Same random seed should produce identical initial weights."""
        torch.manual_seed(123)
        encoder_1 = HGTEncoder(
            **small_encoder_config,
            node_types=mini_node_types,
            edge_categories=mini_edge_categories,
        )

        torch.manual_seed(123)
        encoder_2 = HGTEncoder(
            **small_encoder_config,
            node_types=mini_node_types,
            edge_categories=mini_edge_categories,
        )

        # All parameters should be identical
        for (n1, p1), (n2, p2) in zip(
            encoder_1.named_parameters(), encoder_2.named_parameters()
        ):
            assert n1 == n2
            assert torch.allclose(p1, p2), f"Parameter {n1} differs"

    def test_backward_pass_deterministic(
        self, small_encoder, sample_graph_dict, mini_node_types
    ):
        """Gradients should be reproducible given same input."""
        small_encoder.eval()
        x_dict, edge_index_dict, edge_attr_dict = sample_graph_dict

        # First backward pass
        x_dict_1 = {nt: x_dict[nt].clone().requires_grad_(True) for nt in mini_node_types}
        small_encoder.zero_grad()
        output_1, _ = small_encoder(x_dict_1, edge_index_dict, edge_attr_dict)
        loss_1 = sum(out.sum() for out in output_1.values())
        loss_1.backward()
        grads_1 = {nt: x_dict_1[nt].grad.clone() for nt in mini_node_types}

        # Second backward pass with same input
        x_dict_2 = {nt: x_dict[nt].clone().requires_grad_(True) for nt in mini_node_types}
        small_encoder.zero_grad()
        output_2, _ = small_encoder(x_dict_2, edge_index_dict, edge_attr_dict)
        loss_2 = sum(out.sum() for out in output_2.values())
        loss_2.backward()
        grads_2 = {nt: x_dict_2[nt].grad.clone() for nt in mini_node_types}

        # Gradients should be identical
        for nt in mini_node_types:
            assert torch.allclose(grads_1[nt], grads_2[nt])


# ============================================================================
# LayerScale Interpretability API Tests (HGT-A1, HGT-A2)
# ============================================================================


class TestLayerScales:
    """Test get_layer_scales() and get_mean_layer_scales() interpretability APIs.

    Covers findings HGT-A1 and HGT-A2: these public interpretability methods
    had zero test coverage.
    """

    # --- HGT-A1: get_layer_scales() ---

    def test_get_layer_scales_returns_expected_keys(self, small_encoder):
        """get_layer_scales() should return dict with 'scales', 'cell_types', 'per_cell_type'."""
        result = small_encoder.get_layer_scales()

        assert isinstance(result, dict)
        assert "scales" in result
        assert "cell_types" in result
        assert "per_cell_type" in result

    def test_get_layer_scales_scales_shape(
        self, small_encoder, mini_node_types, small_encoder_config
    ):
        """'scales' tensor should have shape [n_layers, n_node_types]."""
        result = small_encoder.get_layer_scales()
        scales = result["scales"]

        n_layers = small_encoder_config["n_layers"]
        n_node_types = len(mini_node_types)

        assert isinstance(scales, torch.Tensor)
        assert scales.shape == (n_layers, n_node_types)

    def test_get_layer_scales_cell_types_matches_encoder(
        self, small_encoder, mini_node_types
    ):
        """'cell_types' list should match the encoder's node_types."""
        result = small_encoder.get_layer_scales()

        assert result["cell_types"] == mini_node_types

    def test_get_layer_scales_per_cell_type_structure(
        self, small_encoder, mini_node_types, small_encoder_config
    ):
        """'per_cell_type' should map each cell type to a [n_layers] tensor."""
        result = small_encoder.get_layer_scales()
        per_cell_type = result["per_cell_type"]
        n_layers = small_encoder_config["n_layers"]

        assert isinstance(per_cell_type, dict)
        assert set(per_cell_type.keys()) == set(mini_node_types)

        for cell_type in mini_node_types:
            tensor = per_cell_type[cell_type]
            assert isinstance(tensor, torch.Tensor)
            assert tensor.shape == (n_layers,)

    def test_get_layer_scales_values_finite(self, small_encoder):
        """All scale values should be finite (no NaN or Inf)."""
        result = small_encoder.get_layer_scales()
        scales = result["scales"]

        assert torch.isfinite(scales).all(), "Scale values contain NaN or Inf"

    def test_get_layer_scales_values_non_negative_at_init(self, small_encoder):
        """At initialization (layer_scale_init=1.0), all scales should be non-negative."""
        result = small_encoder.get_layer_scales()
        scales = result["scales"]

        assert (scales >= 0).all(), "Initial scale values should be non-negative"

    def test_get_layer_scales_init_value(
        self, small_encoder_config, mini_node_types, mini_edge_categories
    ):
        """Scale values should match layer_scale_init at initialization."""
        init_val = 0.5
        encoder = HGTEncoder(
            **small_encoder_config,
            node_types=mini_node_types,
            edge_categories=mini_edge_categories,
            layer_scale_init=init_val,
        )

        result = encoder.get_layer_scales()
        scales = result["scales"]

        expected = torch.full_like(scales, init_val)
        assert torch.allclose(scales, expected), (
            f"Expected all scales to be {init_val}, got {scales}"
        )

    def test_get_layer_scales_per_cell_type_consistent_with_scales(
        self, small_encoder, mini_node_types
    ):
        """'per_cell_type' values should match corresponding columns of 'scales'."""
        result = small_encoder.get_layer_scales()
        scales = result["scales"]
        per_cell_type = result["per_cell_type"]

        for idx, cell_type in enumerate(mini_node_types):
            expected = scales[:, idx]
            actual = per_cell_type[cell_type]
            assert torch.allclose(actual, expected), (
                f"per_cell_type[{cell_type}] does not match scales[:, {idx}]"
            )

    def test_get_layer_scales_detached(self, small_encoder):
        """Returned scales should be detached (no grad tracking)."""
        result = small_encoder.get_layer_scales()
        scales = result["scales"]

        assert not scales.requires_grad, "Returned scales should be detached from graph"

    # --- HGT-A2: get_mean_layer_scales() ---

    def test_get_mean_layer_scales_returns_dict_of_floats(
        self, small_encoder, mini_node_types
    ):
        """get_mean_layer_scales() should return dict mapping cell type to float."""
        result = small_encoder.get_mean_layer_scales()

        assert isinstance(result, dict)
        assert set(result.keys()) == set(mini_node_types)

        for cell_type, value in result.items():
            assert isinstance(value, float), (
                f"Expected float for {cell_type}, got {type(value)}"
            )

    def test_get_mean_layer_scales_values_finite(self, small_encoder):
        """All mean scale values should be finite."""
        result = small_encoder.get_mean_layer_scales()

        for cell_type, value in result.items():
            assert not (value != value), f"NaN mean scale for {cell_type}"  # NaN check
            assert abs(value) != float("inf"), f"Inf mean scale for {cell_type}"

    def test_get_mean_layer_scales_values_non_negative_at_init(self, small_encoder):
        """At initialization, mean scales should be non-negative."""
        result = small_encoder.get_mean_layer_scales()

        for cell_type, value in result.items():
            assert value >= 0, f"Negative mean scale for {cell_type}: {value}"

    def test_get_mean_layer_scales_consistent_with_get_layer_scales(
        self, small_encoder, mini_node_types
    ):
        """Mean scales should equal the mean across layers from get_layer_scales()."""
        scales_info = small_encoder.get_layer_scales()
        mean_scales = small_encoder.get_mean_layer_scales()

        scales_tensor = scales_info["scales"]  # [n_layers, n_node_types]
        expected_means = scales_tensor.mean(dim=0)  # [n_node_types]

        for idx, cell_type in enumerate(mini_node_types):
            assert abs(mean_scales[cell_type] - expected_means[idx].item()) < 1e-6, (
                f"Mean scale mismatch for {cell_type}: "
                f"got {mean_scales[cell_type]}, expected {expected_means[idx].item()}"
            )

    def test_get_mean_layer_scales_init_value(
        self, small_encoder_config, mini_node_types, mini_edge_categories
    ):
        """At initialization, mean scales should equal layer_scale_init."""
        init_val = 0.7
        encoder = HGTEncoder(
            **small_encoder_config,
            node_types=mini_node_types,
            edge_categories=mini_edge_categories,
            layer_scale_init=init_val,
        )

        result = encoder.get_mean_layer_scales()

        for cell_type, value in result.items():
            assert abs(value - init_val) < 1e-6, (
                f"Expected mean scale {init_val} for {cell_type}, got {value}"
            )

    # --- Batched encoder delegates correctly ---

    def test_batched_encoder_get_layer_scales(
        self, batched_encoder, mini_node_types, small_encoder_config
    ):
        """HGTEncoderBatched.get_layer_scales() should delegate to inner encoder."""
        result = batched_encoder.get_layer_scales()

        assert isinstance(result, dict)
        assert "scales" in result
        assert "cell_types" in result
        assert "per_cell_type" in result
        assert result["cell_types"] == mini_node_types
        assert result["scales"].shape == (
            small_encoder_config["n_layers"],
            len(mini_node_types),
        )

    def test_batched_encoder_get_mean_layer_scales(
        self, batched_encoder, mini_node_types
    ):
        """HGTEncoderBatched.get_mean_layer_scales() should delegate to inner encoder."""
        result = batched_encoder.get_mean_layer_scales()

        assert isinstance(result, dict)
        assert set(result.keys()) == set(mini_node_types)
        for value in result.values():
            assert isinstance(value, float)

    # --- Scales change after training step ---

    def test_layer_scales_change_after_backward(
        self, small_encoder, sample_graph_dict
    ):
        """LayerScale values should change after a training step."""
        scales_before = small_encoder.get_layer_scales()["scales"].clone()

        x_dict, edge_index_dict, edge_attr_dict = sample_graph_dict
        output_dict, _ = small_encoder(x_dict, edge_index_dict, edge_attr_dict)
        loss = sum(out.sum() for out in output_dict.values())
        loss.backward()

        # Simulate optimizer step on layer_scales
        with torch.no_grad():
            for param in small_encoder.layer_scales:
                if param.grad is not None:
                    param -= 0.01 * param.grad

        scales_after = small_encoder.get_layer_scales()["scales"]

        assert not torch.allclose(scales_before, scales_after), (
            "LayerScale values should change after a training step"
        )
