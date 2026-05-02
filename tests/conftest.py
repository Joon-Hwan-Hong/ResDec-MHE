"""
Shared test fixtures for cognitive resilience model tests.
"""

import sys
import types
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Worktree Root + sys.path bootstrap
# ─────────────────────────────────────────────────────────────────────────────
# Single authoritative worktree root constant. All tests should resolve paths
# from this rather than computing parents[N] indices in each test file.
WORKTREE_ROOT: Path = Path(__file__).resolve().parent.parent

# Ensure worktree root is on sys.path so every test can `from src...` /
# `from scripts...` regardless of pytest's invocation cwd. Eliminates the
# repeated `if str(_WORKTREE_ROOT) not in sys.path: sys.path.insert(...)`
# boilerplate previously copy-pasted across 28+ test files.
if str(WORKTREE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKTREE_ROOT))


# ─────────────────────────────────────────────────────────────────────────────
# Matplotlib Agg Backend (headless test runs)
# ─────────────────────────────────────────────────────────────────────────────
# All visualization tests need a non-interactive backend. Set before pyplot
# is imported anywhere in the suite. Centralised here so per-file `import
# matplotlib; matplotlib.use("Agg")` blocks (24+ duplicates) become unneeded.
import matplotlib  # noqa: E402

matplotlib.use("Agg")


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

from src.data.constants import CELL_TYPE_ORDER, ALL_EDGE_TYPES, sanitize_key, N_CELL_TYPES, N_REGIONS


# ─────────────────────────────────────────────────────────────────────────────
# Factory Functions
# ─────────────────────────────────────────────────────────────────────────────


N_EDGE_TYPES = len(ALL_EDGE_TYPES)


def _make_edge_tensors(batch_size, n_edges=5, n_cell_types=N_CELL_TYPES,
                       n_edge_types=N_EDGE_TYPES, device=None):
    """Create flat edge tensors with batch-offset node indices for testing.

    Returns (ccc_edge_index, ccc_edge_type, ccc_edge_attr) in the flat
    concatenated format expected by HGTEncoderTensor and
    CognitiveResilienceModel.
    """
    src_parts, dst_parts, type_parts = [], [], []
    for b in range(batch_size):
        offset = b * n_cell_types
        src_parts.append(torch.randint(0, n_cell_types, (n_edges,)) + offset)
        dst_parts.append(torch.randint(0, n_cell_types, (n_edges,)) + offset)
        type_parts.append(torch.randint(0, n_edge_types, (n_edges,)))
    edge_index = torch.stack([torch.cat(src_parts), torch.cat(dst_parts)])  # [2, B*n_edges]
    edge_type = torch.cat(type_parts)  # [B*n_edges]
    edge_attr = torch.rand(batch_size * n_edges, 1)  # [B*n_edges, 1]
    if device is not None:
        edge_index = edge_index.to(device)
        edge_type = edge_type.to(device)
        edge_attr = edge_attr.to(device)
    return edge_index, edge_type, edge_attr


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


@pytest.fixture(scope="session")
def worktree_root() -> Path:
    """Single authoritative worktree root for path resolution.

    Resolves to the parent of tests/. Use this anywhere a test needs to
    reference configs/, data/, scripts/, etc. — never compute
    `Path(__file__).resolve().parents[N]` per test.
    """
    return WORKTREE_ROOT


@pytest.fixture
def project_root(worktree_root) -> Path:
    """Alias for worktree_root (legacy fixture name).

    Prefer `worktree_root` for new tests.
    """
    return worktree_root


@pytest.fixture
def test_data_dir(worktree_root) -> Path:
    """Get test data directory."""
    return worktree_root / "tests" / "data"


@pytest.fixture(scope="session")
def default_config_path(worktree_root) -> Path:
    """Path to configs/default.yaml resolved from the worktree root.

    Tests previously used the literal `OmegaConf.load("configs/default.yaml")`
    pattern, which only works when pytest is invoked from the worktree root.
    """
    return worktree_root / "configs" / "default.yaml"


# ─────────────────────────────────────────────────────────────────────────────
# Canonical Dummy Batch Factory
# ─────────────────────────────────────────────────────────────────────────────


def _build_canonical_batch(
    batch_size: int = 2,
    n_genes: int = 4785,
    n_ct: int = N_CELL_TYPES,
    n_regions: int = N_REGIONS,
    cells_per_ct: int = 10,
    edges_per_subj: int = 50,
    device: torch.device | str | None = None,
    seed: int = 0,
    include_subject_ids: bool = False,
    pfc_only: bool = True,
) -> dict:
    """Construct a minimal batch matching the model's input contract.

    Replaces 4 duplicate `_make_dummy_batch` / `_build_dummy_train_batch`
    helpers across:
        - tests/unit/models/test_full_model_cf_split.py
        - tests/unit/models/resdec_head/test_encoder_integration.py
        - tests/unit/training/test_resdec_lightning_module.py
        - tests/unit/training/test_resdec_lightning_module_aug_u.py

    Parameters
    ----------
    batch_size:
        Number of subjects in the batch.
    n_genes:
        Number of genes (HVG + L-R). Default 4785 matches canonical config.
    n_ct:
        Number of cell types. Default 31 from CELL_TYPE_ORDER.
    n_regions:
        Number of brain regions. Default 6 from N_REGIONS.
    cells_per_ct:
        Number of cells per cell type per subject (uniform).
    edges_per_subj:
        Number of CCC edges emitted per subject.
    device:
        Device on which output tensors live.
    seed:
        RNG seed for reproducible batch construction.
    include_subject_ids:
        If True, include `subject_ids: list[str]` field (used by aug-U tests).
    pfc_only:
        If True, region_mask is PFC-only [True, False, ..., False]; if False,
        all regions are flagged.
    """
    if device is None:
        device = torch.device("cpu")
    rng = torch.Generator(device="cpu").manual_seed(seed)

    region_mask = torch.zeros(batch_size, n_regions, dtype=torch.bool)
    region_mask[:, 0] = True
    if not pfc_only:
        region_mask[:, :] = True

    region_pseudobulk = torch.randn(
        batch_size, n_regions, n_ct, n_genes, generator=rng,
    )
    region_pseudobulk = region_pseudobulk * region_mask.float().unsqueeze(-1).unsqueeze(-1)

    total_edges = batch_size * edges_per_subj
    if total_edges > 0:
        ccc_edge_index = torch.randint(0, n_ct, (2, total_edges), generator=rng)
        ccc_edge_type = torch.randint(0, 5, (total_edges,), generator=rng)
        ccc_edge_attr = torch.rand(total_edges, 1, generator=rng)
    else:
        ccc_edge_index = torch.zeros(2, 0, dtype=torch.long)
        ccc_edge_type = torch.zeros(0, dtype=torch.long)
        ccc_edge_attr = torch.zeros(0, 1)

    cells_per_subject = cells_per_ct * n_ct
    total_cells = batch_size * cells_per_subject
    cell_data = torch.randn(total_cells, n_genes, generator=rng)
    offsets_per_subj = torch.arange(
        0, cells_per_subject + 1, cells_per_ct, dtype=torch.long,
    )
    subj_offsets = torch.arange(batch_size, dtype=torch.long) * cells_per_subject
    cell_offsets = subj_offsets.unsqueeze(1) + offsets_per_subj.unsqueeze(0)

    batch = {
        "region_pseudobulk": region_pseudobulk.to(device),
        "region_mask": region_mask.to(device),
        "ccc_edge_index": ccc_edge_index.to(device),
        "ccc_edge_type": ccc_edge_type.to(device),
        "ccc_edge_attr": ccc_edge_attr.to(device),
        "cell_type_mask": torch.ones(batch_size, n_ct, dtype=torch.bool).to(device),
        "cell_data": cell_data.to(device),
        "cell_offsets": cell_offsets.to(device),
        "pathology": torch.randn(batch_size, 3, generator=rng).to(device),
        "cognition": torch.randn(batch_size, 1, generator=rng).to(device),
    }
    if include_subject_ids:
        batch["subject_ids"] = [f"sid_{i:03d}" for i in range(batch_size)]
    return batch


@pytest.fixture
def make_canonical_batch():
    """Factory fixture: returns the canonical batch builder.

    Usage::

        def test_X(make_canonical_batch):
            batch = make_canonical_batch(batch_size=2, n_genes=64)
    """
    return _build_canonical_batch


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


@pytest.fixture(autouse=True)
def close_matplotlib_figures():
    """Close all matplotlib figures after each test.

    Visualization tests previously each defined their own autouse fixture
    (`@pytest.fixture(autouse=True) def cleanup(): yield; plt.close("all")`).
    Centralised here so the cleanup happens for every test in the suite — no
    fixture leakage if a non-vis test accidentally creates a figure.
    """
    yield
    try:
        import matplotlib.pyplot as plt
        plt.close("all")
    except Exception:
        # plt may not have been imported in this test process; benign.
        pass


@pytest.fixture(autouse=True)
def clear_pyro_param_store():
    """Clear Pyro's global param store and reset settings before every test.

    Pyro uses a global param store that persists across tests. If a test
    creates Pyro parameters (e.g., via SVI, AutoDiagonalNormal) and another
    test later creates a new BayesianPredictionHead with the same parameter
    names, the stale entries collide — the new guide sees nn.Parameter objects
    instead of Pyro's unconstrained params, causing
    'Parameter has no attribute unconstrained' errors.

    Also resets module_local_params to False (Pyro default). Some tests
    import scripts.training.train which calls pyro.settings.set(module_local_params=True)
    at module scope. This global setting persists across tests and breaks SVI
    tests that expect parameters in the global param store.

    Clearing before (not after) each test ensures a clean slate regardless
    of whether the previous test cleaned up after itself.
    """
    import pyro
    pyro.clear_param_store()
    pyro.settings.set(module_local_params=False)