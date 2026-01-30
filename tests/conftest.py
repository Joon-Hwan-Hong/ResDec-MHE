"""
Shared test fixtures for cognitive resilience model tests.
"""

import pytest
import torch
import numpy as np
from pathlib import Path


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
    return 31


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
    mask = torch.zeros(batch_size, 6, dtype=torch.bool)
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
    """Automatically set seeds before each test."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)