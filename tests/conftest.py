"""
Shared test fixtures for cognitive resilience model tests.
"""

import sys
import types

# ─────────────────────────────────────────────────────────────────────────────
# Torchvision Mock (broken C extension workaround)
# ─────────────────────────────────────────────────────────────────────────────
# Lightning imports torchmetrics which imports torchvision, but torchvision._C.so
# has undefined symbols in this environment. Since tests don't need torchvision,
# mock it before lightning is imported. Applied at top-level conftest so all test
# directories share one copy (previously duplicated in smoke/ and unit/training/).


class _MockModule(types.ModuleType):
    """Mock module that returns sub-mocks for any attribute access."""

    def __init__(self, name):
        super().__init__(name)
        self.__path__ = []
        self.__file__ = "<mock>"
        self.__version__ = "0.25.0"
        self.__all__ = []

    def __getattr__(self, name):
        sub = _MockModule(f"{self.__name__}.{name}")
        setattr(self, name, sub)
        return sub

    def __call__(self, *args, **kwargs):
        return None


if "torchvision" not in sys.modules:
    _tv_names = [
        "torchvision", "torchvision._meta_registrations",
        "torchvision.datasets", "torchvision.io", "torchvision.models",
        "torchvision.ops", "torchvision.transforms", "torchvision.utils",
        "torchvision.transforms.functional", "torchvision.extension",
    ]
    for mod_name in _tv_names:
        sys.modules[mod_name] = _MockModule(mod_name)
    sys.modules["torchvision"].extension._has_ops = lambda: False

# ─────────────────────────────────────────────────────────────────────────────

import pytest
import torch
import numpy as np
from pathlib import Path

from src.data.constants import CELL_TYPE_ORDER, ALL_EDGE_TYPES, sanitize_key, N_CELL_TYPES, N_REGIONS


# ─────────────────────────────────────────────────────────────────────────────
# Factory Functions
# ─────────────────────────────────────────────────────────────────────────────


N_EDGE_TYPES = len(ALL_EDGE_TYPES)


def _make_edge_tensors(batch_size, n_edges=5, n_cell_types=N_CELL_TYPES,
                       n_edge_types=N_EDGE_TYPES, device=None):
    """Create batched edge tensors for testing.

    Returns (ccc_edge_index, ccc_edge_type, ccc_edge_attr, ccc_edge_counts)
    in the padded tensor format expected by HGTEncoderTensor and
    CognitiveResilienceModel.
    """
    edge_index = torch.randint(0, n_cell_types, (batch_size, 2, n_edges))
    edge_type = torch.randint(0, n_edge_types, (batch_size, n_edges))
    edge_attr = torch.rand(batch_size, n_edges, 1)
    edge_counts = torch.full((batch_size,), n_edges, dtype=torch.long)
    if device is not None:
        edge_index = edge_index.to(device)
        edge_type = edge_type.to(device)
        edge_attr = edge_attr.to(device)
        edge_counts = edge_counts.to(device)
    return edge_index, edge_type, edge_attr, edge_counts


# ─────────────────────────────────────────────────────────────────────────────
# Factory Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def make_edge_tensors():
    """Factory fixture: returns _make_edge_tensors callable."""
    return _make_edge_tensors


@pytest.fixture
def small_model_config():
    """Small model configuration for testing."""
    return {
        'n_genes': 50, 'n_cell_types': N_CELL_TYPES, 'd_embed': 32,
        'd_fused': 32, 'd_cond': 16, 'n_regions': N_REGIONS,
        'n_hgt_layers': 1, 'n_hgt_heads': 4, 'n_isab_layers': 1,
        'n_inducing_points': 4, 'n_attention_heads': 4,
        'd_head_hidden': 16, 'dropout': 0.0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Device and Hardware Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def device():
    """Get available device (CUDA if available, else CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def seed():
    """Default seed for reproducibility."""
    return 42


# ─────────────────────────────────────────────────────────────────────────────
# Dimension Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def batch_size():
    """Default batch size for testing."""
    return 4


@pytest.fixture
def n_genes():
    """Number of genes (HVGs + L-R genes)."""
    return 3000


@pytest.fixture
def n_cell_types():
    """Number of Allen ABC cell types."""
    return N_CELL_TYPES


@pytest.fixture
def d_embed():
    """Embedding dimension."""
    return 128


@pytest.fixture
def n_heads():
    """Number of attention heads."""
    return 4


@pytest.fixture
def max_cells():
    """Maximum cells per cell type."""
    return 100  # Smaller for faster tests


# ─────────────────────────────────────────────────────────────────────────────
# Dummy Data Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def dummy_pseudobulk(batch_size, n_cell_types, n_genes):
    """Dummy pseudobulk expression data."""
    return torch.randn(batch_size, n_cell_types, n_genes)


@pytest.fixture
def dummy_pathology(batch_size):
    """Dummy pathology scores [gpath, amylsqrt, tangsqrt]."""
    return torch.rand(batch_size, 3)


@pytest.fixture
def dummy_cognition(batch_size):
    """Dummy cognition target."""
    return torch.randn(batch_size, 1)


@pytest.fixture
def dummy_region_mask(batch_size):
    """Dummy region mask (only PFC available)."""
    mask = torch.zeros(batch_size, N_REGIONS, dtype=torch.bool)
    mask[:, 0] = True  # PFC always available
    return mask


# ─────────────────────────────────────────────────────────────────────────────
# Graph Fixtures (for HGT)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def dummy_edge_index(n_cell_types):
    """Dummy edge index for a simple CCC graph."""
    # Create some random edges between cell types
    n_edges = 50
    src = torch.randint(0, n_cell_types, (n_edges,))
    dst = torch.randint(0, n_cell_types, (n_edges,))
    return torch.stack([src, dst], dim=0)


@pytest.fixture
def dummy_edge_type(n_cell_types):
    """Dummy edge types (5 CellChatDB categories)."""
    n_edges = 50
    return torch.randint(0, 5, (n_edges,))


@pytest.fixture
def dummy_edge_attr(n_cell_types):
    """Dummy edge attributes (LIANA+ scores)."""
    n_edges = 50
    return torch.rand(n_edges, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Configuration Fixtures
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# Path Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def project_root():
    """Get project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture
def test_data_dir(project_root):
    """Get test data directory."""
    return project_root / "tests" / "data"


# ─────────────────────────────────────────────────────────────────────────────
# Seeding and Reproducibility
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def set_random_seeds(seed):
    """Reset all RNG state before every test for reproducibility.

    Autouse=True means this runs before EVERY test without explicit request.
    Seeds Python random, NumPy, PyTorch CPU, and PyTorch CUDA. This ensures
    tests using random data (torch.randn, np.random) produce identical results
    across runs, making flaky test failures reproducible.
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)