"""
Integration tests for data pipeline to CognitiveResilienceModel.

Tests verify that:
- Collated batch keys match what CognitiveResilienceModel.forward() expects
- Tensor shapes from data pipeline are compatible with model input shapes
- Single-region, multi-region, and mixed batches work correctly
- Model output has expected keys and shapes
- DataLoader iteration works with the model

These tests use synthetic/mock data to work without actual ROSMAP data files.
"""

import pytest
import torch
from torch.utils.data import Dataset, DataLoader

from src.data.constants import (
    N_CELL_TYPES,
    N_EDGE_TYPES,
    N_REGIONS,
    CELL_TYPE_ORDER,
    ALL_EDGE_TYPES,
)
from src.data.collate import (
    collate_fn,
    collate_for_hgt,
    collate_for_hgt_multiregion,
    create_dataloader,
)
from src.models.full_model import CognitiveResilienceModel


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def n_genes():
    """Number of genes for test data."""
    return 100


@pytest.fixture
def max_cells():
    """Maximum cells per cell type for test data."""
    return 50


@pytest.fixture
def model_config(n_genes):
    """Configuration for creating test models.

    Returns configuration matching test data dimensions.
    """
    return {
        "n_genes": n_genes,
        "n_cell_types": N_CELL_TYPES,
        "d_embed": 32,
        "d_fused": 32,
        "d_cond": 16,
        "n_regions": N_REGIONS,
        "n_hgt_layers": 1,
        "n_hgt_heads": 2,
        "n_isab_layers": 1,
        "n_inducing_points": 8,
        "n_attention_heads": 2,
        "d_head_hidden": 16,
        "dropout": 0.0,  # Disable dropout for deterministic testing
        "use_bayesian_head": False,
    }


@pytest.fixture
def model(model_config):
    """Create a CognitiveResilienceModel for testing."""
    return CognitiveResilienceModel(**model_config)


def create_mock_dataset_sample(
    n_genes: int = 100,
    max_cells: int = 50,
    n_edges: int = 20,
    subject_id: str = "TEST_SUBJECT",
    include_cell_type_order: bool = True,
) -> dict:
    """Create a sample matching CognitiveResilienceDataset output format.

    Mirrors real dataset schema: uses production N_CELL_TYPES/N_REGIONS from constants.
    Divergences from real data: uses random tensors (real data is sparse, non-negative
    expression); all cell_type_mask/cell_mask entries are True (real data has masked types);
    edge indices are uniform random (real CCC edges have structure).

    If the real dataset schema changes (new keys, shape changes), update this factory
    and all tests that use it.

    Args:
        n_genes: Number of genes
        max_cells: Maximum cells per cell type
        n_edges: Number of CCC edges
        subject_id: Subject identifier
        include_cell_type_order: Whether to include cell_type_order key

    Returns:
        Dictionary matching CognitiveResilienceDataset.__getitem__() output
    """
    sample = {
        "subject_id": subject_id,
        "pseudobulk": torch.randn(N_CELL_TYPES, n_genes),
        "cell_type_mask": torch.ones(N_CELL_TYPES, dtype=torch.bool),
        "cell_counts": torch.randint(10, 100, (N_CELL_TYPES,)),
        "cells": torch.randn(N_CELL_TYPES, max_cells, n_genes),
        "cell_mask": torch.ones(N_CELL_TYPES, max_cells, dtype=torch.bool),
        "ccc_edge_index": torch.randint(0, N_CELL_TYPES, (2, n_edges)),
        "ccc_edge_type": torch.randint(0, N_EDGE_TYPES, (n_edges,)),
        "ccc_edge_attr": torch.rand(n_edges, 1),
        "pathology": torch.rand(3),
        "cognition": torch.randn(1),
        "region_mask": torch.ones(N_REGIONS, dtype=torch.bool),
    }
    if include_cell_type_order:
        sample["cell_type_order"] = CELL_TYPE_ORDER
    return sample


def create_single_region_sample(
    n_genes: int = 100,
    max_cells: int = 50,
    n_edges: int = 20,
    subject_id: str = "SINGLE_REGION_SUBJECT",
) -> dict:
    """Create a sample with only PFC region available (single region)."""
    sample = create_mock_dataset_sample(
        n_genes=n_genes,
        max_cells=max_cells,
        n_edges=n_edges,
        subject_id=subject_id,
    )
    # Only first region (PFC) is available
    region_mask = torch.zeros(N_REGIONS, dtype=torch.bool)
    region_mask[0] = True
    sample["region_mask"] = region_mask
    return sample


def create_multi_region_sample(
    n_genes: int = 100,
    max_cells: int = 50,
    n_edges: int = 20,
    subject_id: str = "MULTI_REGION_SUBJECT",
    n_available_regions: int = 4,
) -> dict:
    """Create a sample with multiple brain regions available.

    Also includes per-region pseudobulk data for collate_for_hgt_multiregion.
    """
    sample = create_mock_dataset_sample(
        n_genes=n_genes,
        max_cells=max_cells,
        n_edges=n_edges,
        subject_id=subject_id,
    )
    # Multiple regions available
    region_mask = torch.zeros(N_REGIONS, dtype=torch.bool)
    region_mask[:n_available_regions] = True
    sample["region_mask"] = region_mask

    # Add per-region pseudobulk data
    available_regions = list(range(n_available_regions))
    sample["available_regions"] = available_regions
    for region_idx in available_regions:
        sample[f"region_{region_idx}_pseudobulk"] = torch.randn(N_CELL_TYPES, n_genes)

    return sample


def convert_collated_batch_to_model_input(
    collated: dict,
    n_genes: int,
) -> dict:
    """Convert collate_fn output to CognitiveResilienceModel.forward() input format.

    The model expects specific keys and shapes that differ slightly from collate output:
    - region_pseudobulk: [B, n_regions, n_cell_types, n_genes]
    - region_mask: [B, n_regions]
    - edge_index_dict_list: List of {(src, rel, dst): [2, n_edges]} per sample
    - edge_attr_dict_list: List of {(src, rel, dst): [n_edges, 1]} per sample
    - cells: [B, n_cell_types, max_cells, n_genes]
    - cell_mask: [B, n_cell_types, max_cells]
    - pathology: [B, 3]
    - cognition: [B, 1] (optional)

    Note: This conversion creates region_pseudobulk from pseudobulk for compatibility,
    as actual multi-region data requires collate_for_hgt_multiregion.
    """
    # Create region_pseudobulk from single pseudobulk (replicate to all regions)
    # In real usage, collate_for_hgt_multiregion provides actual per-region data
    pseudobulk = collated["pseudobulk"]  # [B, n_cell_types, n_genes]
    region_pseudobulk = pseudobulk.unsqueeze(1).expand(-1, N_REGIONS, -1, -1).clone()

    result = {
        "region_pseudobulk": region_pseudobulk,
        "region_mask": collated["region_mask"],
        "cells": collated["cells"],
        "cell_mask": collated["cell_mask"],
        "pathology": collated["pathology"],
        "cognition": collated["cognition"],
    }

    # Pass through edge dict lists if available (from collate_for_hgt)
    if "edge_index_dict_list" in collated:
        result["edge_index_dict_list"] = collated["edge_index_dict_list"]
        result["edge_attr_dict_list"] = collated["edge_attr_dict_list"]

    return result


def convert_multiregion_collated_to_model_input(collated: dict) -> dict:
    """Convert collate_for_hgt_multiregion output to model input format.

    collate_for_hgt_multiregion provides region_pseudobulk directly when
    multiregion data is detected. For single-region batches, we fall back
    to creating region_pseudobulk from the base pseudobulk.
    """
    # Handle case where region_pseudobulk wasn't created (single-region batch
    # without multiregion data detected)
    if "region_pseudobulk" in collated:
        region_pseudobulk = collated["region_pseudobulk"]
    else:
        # Fall back to creating region_pseudobulk from base pseudobulk
        pseudobulk = collated["pseudobulk"]  # [B, n_cell_types, n_genes]
        region_pseudobulk = pseudobulk.unsqueeze(1).expand(-1, N_REGIONS, -1, -1).clone()

    result = {
        "region_pseudobulk": region_pseudobulk,
        "region_mask": collated["region_mask"],
        "cells": collated["cells"],
        "cell_mask": collated["cell_mask"],
        "pathology": collated["pathology"],
        "cognition": collated["cognition"],
    }

    # Pass through edge dict lists if available (from collate_for_hgt / collate_for_hgt_multiregion)
    if "edge_index_dict_list" in collated:
        result["edge_index_dict_list"] = collated["edge_index_dict_list"]
        result["edge_attr_dict_list"] = collated["edge_attr_dict_list"]

    return result


class MockROSMAPDataset(Dataset):
    """Mock dataset mimicking ROSMAPDataset output format.

    Generates synthetic data for testing without actual ROSMAP data files.
    """

    def __init__(
        self,
        n_samples: int = 10,
        n_genes: int = 100,
        max_cells: int = 50,
        n_edges: int = 20,
        mode: str = "single_region",  # "single_region", "multi_region", "mixed"
    ):
        self.n_samples = n_samples
        self.n_genes = n_genes
        self.max_cells = max_cells
        self.n_edges = n_edges
        self.mode = mode

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> dict:
        subject_id = f"SUBJECT_{idx:04d}"

        if self.mode == "single_region":
            return create_single_region_sample(
                n_genes=self.n_genes,
                max_cells=self.max_cells,
                n_edges=self.n_edges,
                subject_id=subject_id,
            )
        elif self.mode == "multi_region":
            return create_multi_region_sample(
                n_genes=self.n_genes,
                max_cells=self.max_cells,
                n_edges=self.n_edges,
                subject_id=subject_id,
                n_available_regions=4,
            )
        else:  # mixed
            if idx % 2 == 0:
                return create_single_region_sample(
                    n_genes=self.n_genes,
                    max_cells=self.max_cells,
                    n_edges=self.n_edges,
                    subject_id=subject_id,
                )
            else:
                return create_multi_region_sample(
                    n_genes=self.n_genes,
                    max_cells=self.max_cells,
                    n_edges=self.n_edges,
                    subject_id=subject_id,
                    n_available_regions=3,
                )


# =============================================================================
# Test Classes
# =============================================================================


class TestCollatedBatchKeysMatchModelInput:
    """Test that keys from collate_fn match what CognitiveResilienceModel.forward() expects."""

    def test_collate_fn_output_contains_required_keys(self, n_genes, max_cells):
        """Verify collate_fn output contains all keys needed for model conversion."""
        batch = [
            create_mock_dataset_sample(n_genes=n_genes, max_cells=max_cells)
            for _ in range(4)
        ]
        collated = collate_fn(batch)

        # Keys expected from collate_fn output
        required_keys = [
            "pseudobulk",
            "region_mask",
            "ccc_edge_index",
            "ccc_edge_attr",
            "cells",
            "cell_mask",
            "pathology",
            "cognition",
            "batch_size",
        ]

        for key in required_keys:
            assert key in collated, f"Missing required key: {key}"

    def test_collate_for_hgt_multiregion_produces_region_pseudobulk(self, n_genes, max_cells):
        """Verify collate_for_hgt_multiregion produces region_pseudobulk key."""
        batch = [
            create_multi_region_sample(n_genes=n_genes, max_cells=max_cells)
            for _ in range(4)
        ]
        collated = collate_for_hgt_multiregion(batch)

        assert "region_pseudobulk" in collated, "Missing region_pseudobulk key"
        assert "region_mask" in collated, "Missing region_mask key"

    def test_model_forward_signature_matches_converted_batch(self, model, n_genes, max_cells):
        """Verify converted batch keys match model.forward() parameters."""
        batch = [
            create_mock_dataset_sample(n_genes=n_genes, max_cells=max_cells)
            for _ in range(4)
        ]
        collated = collate_fn(batch)
        model_input = convert_collated_batch_to_model_input(collated, n_genes)

        # Model forward should accept these exact keys
        required_forward_params = [
            "region_pseudobulk",
            "region_mask",
            "cells",
            "cell_mask",
            "pathology",
        ]

        for param in required_forward_params:
            assert param in model_input, f"Missing model input: {param}"

        # Should be able to call forward without errors
        output = model(**model_input)
        assert "mean" in output


class TestCollatedBatchShapesCompatible:
    """Test that tensor shapes from data pipeline are compatible with model input shapes."""

    def test_pseudobulk_shape(self, n_genes, max_cells):
        """Pseudobulk should have shape [B, n_cell_types, n_genes]."""
        batch_size = 4
        batch = [
            create_mock_dataset_sample(n_genes=n_genes, max_cells=max_cells)
            for _ in range(batch_size)
        ]
        collated = collate_fn(batch)

        assert collated["pseudobulk"].shape == (batch_size, N_CELL_TYPES, n_genes)

    def test_region_pseudobulk_shape_after_conversion(self, n_genes, max_cells):
        """Region pseudobulk should have shape [B, n_regions, n_cell_types, n_genes]."""
        batch_size = 4
        batch = [
            create_multi_region_sample(n_genes=n_genes, max_cells=max_cells)
            for _ in range(batch_size)
        ]
        collated = collate_for_hgt_multiregion(batch)

        assert collated["region_pseudobulk"].shape == (
            batch_size, N_REGIONS, N_CELL_TYPES, n_genes
        )

    def test_cells_shape(self, n_genes, max_cells):
        """Cells should have shape [B, n_cell_types, max_cells, n_genes]."""
        batch_size = 4
        batch = [
            create_mock_dataset_sample(n_genes=n_genes, max_cells=max_cells)
            for _ in range(batch_size)
        ]
        collated = collate_fn(batch)

        assert collated["cells"].shape == (batch_size, N_CELL_TYPES, max_cells, n_genes)

    def test_cell_mask_shape(self, n_genes, max_cells):
        """Cell mask should have shape [B, n_cell_types, max_cells]."""
        batch_size = 4
        batch = [
            create_mock_dataset_sample(n_genes=n_genes, max_cells=max_cells)
            for _ in range(batch_size)
        ]
        collated = collate_fn(batch)

        assert collated["cell_mask"].shape == (batch_size, N_CELL_TYPES, max_cells)
        assert collated["cell_mask"].dtype == torch.bool

    def test_region_mask_shape(self, n_genes, max_cells):
        """Region mask should have shape [B, n_regions]."""
        batch_size = 4
        batch = [
            create_mock_dataset_sample(n_genes=n_genes, max_cells=max_cells)
            for _ in range(batch_size)
        ]
        collated = collate_fn(batch)

        assert collated["region_mask"].shape == (batch_size, N_REGIONS)
        assert collated["region_mask"].dtype == torch.bool

    def test_pathology_shape(self, n_genes, max_cells):
        """Pathology should have shape [B, 3]."""
        batch_size = 4
        batch = [
            create_mock_dataset_sample(n_genes=n_genes, max_cells=max_cells)
            for _ in range(batch_size)
        ]
        collated = collate_fn(batch)

        assert collated["pathology"].shape == (batch_size, 3)

    def test_cognition_shape(self, n_genes, max_cells):
        """Cognition should have shape [B, 1]."""
        batch_size = 4
        batch = [
            create_mock_dataset_sample(n_genes=n_genes, max_cells=max_cells)
            for _ in range(batch_size)
        ]
        collated = collate_fn(batch)

        assert collated["cognition"].shape == (batch_size, 1)

    def test_edge_index_shape(self, n_genes, max_cells):
        """Edge index should have shape [2, total_edges]."""
        batch = [
            create_mock_dataset_sample(n_genes=n_genes, max_cells=max_cells, n_edges=10)
            for _ in range(4)
        ]
        collated = collate_fn(batch)

        assert collated["ccc_edge_index"].shape[0] == 2
        # Total edges depends on how many edges per sample


class TestSingleRegionBatchThroughModel:
    """Test that a batch with single-region subjects (PFC only) passes through the model."""

    def test_single_region_batch_forward_pass(self, model, n_genes, max_cells):
        """Single-region batch should pass through model without errors."""
        batch = [
            create_single_region_sample(n_genes=n_genes, max_cells=max_cells)
            for _ in range(4)
        ]
        collated = collate_fn(batch)
        model_input = convert_collated_batch_to_model_input(collated, n_genes)

        output = model(**model_input)

        assert "mean" in output
        assert torch.isfinite(output["mean"]).all()

    def test_single_region_batch_output_shape(self, model, n_genes, max_cells):
        """Single-region batch output should have correct shape."""
        batch_size = 4
        batch = [
            create_single_region_sample(n_genes=n_genes, max_cells=max_cells)
            for _ in range(batch_size)
        ]
        collated = collate_fn(batch)
        model_input = convert_collated_batch_to_model_input(collated, n_genes)

        output = model(**model_input)

        assert output["mean"].shape == (batch_size, 1)

    def test_single_region_mask_correctly_set(self, n_genes, max_cells):
        """Single-region samples should have only first region in mask."""
        batch = [
            create_single_region_sample(n_genes=n_genes, max_cells=max_cells)
            for _ in range(4)
        ]
        collated = collate_fn(batch)

        # Only first region should be True
        region_mask = collated["region_mask"]
        assert region_mask[:, 0].all()
        assert not region_mask[:, 1:].any()


class TestMultiRegionBatchThroughModel:
    """Test that a batch with multi-region subjects passes through the model."""

    def test_multi_region_batch_forward_pass(self, model, n_genes, max_cells):
        """Multi-region batch should pass through model without errors."""
        batch = [
            create_multi_region_sample(n_genes=n_genes, max_cells=max_cells)
            for _ in range(4)
        ]
        collated = collate_for_hgt_multiregion(batch)
        model_input = convert_multiregion_collated_to_model_input(collated)

        output = model(**model_input)

        assert "mean" in output
        assert torch.isfinite(output["mean"]).all()

    def test_multi_region_batch_output_shape(self, model, n_genes, max_cells):
        """Multi-region batch output should have correct shape."""
        batch_size = 4
        batch = [
            create_multi_region_sample(n_genes=n_genes, max_cells=max_cells)
            for _ in range(batch_size)
        ]
        collated = collate_for_hgt_multiregion(batch)
        model_input = convert_multiregion_collated_to_model_input(collated)

        output = model(**model_input)

        assert output["mean"].shape == (batch_size, 1)

    def test_multi_region_mask_correctly_set(self, n_genes, max_cells):
        """Multi-region samples should have multiple regions in mask."""
        n_available = 4
        batch = [
            create_multi_region_sample(
                n_genes=n_genes,
                max_cells=max_cells,
                n_available_regions=n_available,
            )
            for _ in range(4)
        ]
        collated = collate_for_hgt_multiregion(batch)

        # First n_available regions should be True
        region_mask = collated["region_mask"]
        assert region_mask[:, :n_available].all()

    def test_multi_region_pseudobulk_has_data(self, n_genes, max_cells):
        """Multi-region pseudobulk should have non-zero data for available regions."""
        batch = [
            create_multi_region_sample(n_genes=n_genes, max_cells=max_cells)
            for _ in range(4)
        ]
        collated = collate_for_hgt_multiregion(batch)

        region_pseudobulk = collated["region_pseudobulk"]
        region_mask = collated["region_mask"]

        # Data for available regions should be non-zero
        for b in range(4):
            for r in range(N_REGIONS):
                if region_mask[b, r]:
                    # At least some genes should have non-zero values
                    assert region_pseudobulk[b, r].abs().sum() > 0


class TestMixedBatchSingleAndMultiRegion:
    """Test that a mixed batch (some single-region, some multi-region) works correctly."""

    def test_mixed_batch_forward_pass(self, model, n_genes, max_cells):
        """Mixed batch should pass through model without errors."""
        # Create batch with alternating single/multi region samples
        batch = []
        for i in range(4):
            if i % 2 == 0:
                batch.append(create_single_region_sample(
                    n_genes=n_genes, max_cells=max_cells
                ))
            else:
                batch.append(create_multi_region_sample(
                    n_genes=n_genes, max_cells=max_cells
                ))

        collated = collate_for_hgt_multiregion(batch)
        model_input = convert_multiregion_collated_to_model_input(collated)

        output = model(**model_input)

        assert "mean" in output
        assert torch.isfinite(output["mean"]).all()

    def test_mixed_batch_different_region_masks(self, n_genes, max_cells):
        """Mixed batch should have different region masks for different samples."""
        batch = []
        for i in range(4):
            if i % 2 == 0:
                batch.append(create_single_region_sample(
                    n_genes=n_genes, max_cells=max_cells
                ))
            else:
                batch.append(create_multi_region_sample(
                    n_genes=n_genes, max_cells=max_cells, n_available_regions=4
                ))

        collated = collate_for_hgt_multiregion(batch)
        region_mask = collated["region_mask"]

        # Check masks differ between single and multi-region samples
        single_region_count = region_mask[0].sum()
        multi_region_count = region_mask[1].sum()

        assert single_region_count == 1, "Single region sample should have 1 region"
        assert multi_region_count > 1, "Multi region sample should have >1 regions"

    def test_mixed_batch_gradient_flow(self, model, n_genes, max_cells):
        """Gradients should flow through model with mixed batch."""
        batch = []
        for i in range(4):
            if i % 2 == 0:
                batch.append(create_single_region_sample(
                    n_genes=n_genes, max_cells=max_cells
                ))
            else:
                batch.append(create_multi_region_sample(
                    n_genes=n_genes, max_cells=max_cells
                ))

        collated = collate_for_hgt_multiregion(batch)
        model_input = convert_multiregion_collated_to_model_input(collated)

        # Enable gradients on inputs
        model_input["region_pseudobulk"].requires_grad_(True)
        model_input["cells"].requires_grad_(True)

        output = model(**model_input)
        loss = output["mean"].sum()
        loss.backward()

        # Check gradients flow to inputs
        assert model_input["region_pseudobulk"].grad is not None
        assert model_input["cells"].grad is not None


class TestModelOutputKeysAndShapes:
    """Test that model output dictionary has expected keys and shapes."""

    def test_deterministic_model_output_keys(self, model_config, n_genes, max_cells):
        """Deterministic model should output 'mean' and 'attention_weights'."""
        model_config = dict(model_config)
        model_config["use_bayesian_head"] = False
        model = CognitiveResilienceModel(**model_config)

        batch = [
            create_mock_dataset_sample(n_genes=n_genes, max_cells=max_cells)
            for _ in range(4)
        ]
        collated = collate_fn(batch)
        model_input = convert_collated_batch_to_model_input(collated, n_genes)

        output = model(**model_input)

        assert "mean" in output
        assert "attention_weights" in output
        assert "std" not in output

    def test_bayesian_model_output_keys(self, model_config, n_genes, max_cells):
        """Bayesian model should output 'mean', 'std', and 'attention_weights'."""
        model_config = dict(model_config)
        model_config["use_bayesian_head"] = True
        model = CognitiveResilienceModel(**model_config)

        batch = [
            create_mock_dataset_sample(n_genes=n_genes, max_cells=max_cells)
            for _ in range(4)
        ]
        collated = collate_fn(batch)
        model_input = convert_collated_batch_to_model_input(collated, n_genes)

        output = model(**model_input)

        assert "mean" in output
        assert "std" in output
        assert "attention_weights" in output

    def test_output_mean_shape(self, model, n_genes, max_cells):
        """Output mean should have shape [B, 1]."""
        batch_size = 4
        batch = [
            create_mock_dataset_sample(n_genes=n_genes, max_cells=max_cells)
            for _ in range(batch_size)
        ]
        collated = collate_fn(batch)
        model_input = convert_collated_batch_to_model_input(collated, n_genes)

        output = model(**model_input)

        assert output["mean"].shape == (batch_size, 1)

    def test_output_attention_weights_shape(self, model_config, n_genes, max_cells):
        """Attention weights should have shape [B, n_heads, n_cell_types]."""
        model = CognitiveResilienceModel(**model_config)
        batch_size = 4

        batch = [
            create_mock_dataset_sample(n_genes=n_genes, max_cells=max_cells)
            for _ in range(batch_size)
        ]
        collated = collate_fn(batch)
        model_input = convert_collated_batch_to_model_input(collated, n_genes)

        output = model(**model_input)

        n_heads = model_config["n_attention_heads"]
        assert output["attention_weights"].shape == (batch_size, n_heads, N_CELL_TYPES)

    def test_output_attention_weights_valid(self, model, n_genes, max_cells):
        """Attention weights should sum to 1 and be non-negative."""
        batch = [
            create_mock_dataset_sample(n_genes=n_genes, max_cells=max_cells)
            for _ in range(4)
        ]
        collated = collate_fn(batch)
        model_input = convert_collated_batch_to_model_input(collated, n_genes)

        output = model(**model_input)
        attention = output["attention_weights"]

        # Non-negative
        assert (attention >= 0).all()

        # Sum to 1 across cell types
        sums = attention.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


class TestDataLoaderIterationWithModel:
    """Test that iterating through a DataLoader and passing to model works."""

    def test_dataloader_with_collate_fn_and_model(self, model, n_genes, max_cells):
        """DataLoader iteration should produce valid model inputs."""
        dataset = MockROSMAPDataset(
            n_samples=12,
            n_genes=n_genes,
            max_cells=max_cells,
            mode="single_region",
        )
        loader = DataLoader(
            dataset,
            batch_size=4,
            collate_fn=collate_fn,
            num_workers=0,
        )

        n_batches = 0
        for batch in loader:
            model_input = convert_collated_batch_to_model_input(batch, n_genes)
            output = model(**model_input)

            assert torch.isfinite(output["mean"]).all()
            n_batches += 1

        assert n_batches == 3  # 12 samples / 4 batch_size

    def test_dataloader_with_hgt_multiregion_collate(self, model, n_genes, max_cells):
        """DataLoader with collate_for_hgt_multiregion should work."""
        dataset = MockROSMAPDataset(
            n_samples=12,
            n_genes=n_genes,
            max_cells=max_cells,
            mode="multi_region",
        )
        loader = DataLoader(
            dataset,
            batch_size=4,
            collate_fn=collate_for_hgt_multiregion,
            num_workers=0,
        )

        for batch in loader:
            model_input = convert_multiregion_collated_to_model_input(batch)
            output = model(**model_input)

            assert torch.isfinite(output["mean"]).all()

    def test_dataloader_with_mixed_samples(self, model, n_genes, max_cells):
        """DataLoader with mixed single/multi region samples should work."""
        dataset = MockROSMAPDataset(
            n_samples=12,
            n_genes=n_genes,
            max_cells=max_cells,
            mode="mixed",
        )
        loader = DataLoader(
            dataset,
            batch_size=4,
            collate_fn=collate_for_hgt_multiregion,
            num_workers=0,
        )

        for batch in loader:
            model_input = convert_multiregion_collated_to_model_input(batch)
            output = model(**model_input)

            assert torch.isfinite(output["mean"]).all()

    def test_create_dataloader_helper_function(self, model, n_genes, max_cells):
        """create_dataloader helper should produce valid model inputs."""
        dataset = MockROSMAPDataset(
            n_samples=8,
            n_genes=n_genes,
            max_cells=max_cells,
            mode="multi_region",
        )

        loader = create_dataloader(
            dataset,
            batch_size=4,
            shuffle=False,
            num_workers=0,
            multiregion=True,
            use_hgt_format=True,
        )

        for batch in loader:
            model_input = convert_multiregion_collated_to_model_input(batch)
            output = model(**model_input)

            assert torch.isfinite(output["mean"]).all()

    def test_dataloader_training_loop_simulation(self, model, n_genes, max_cells):
        """Simulate a training loop with DataLoader and model."""
        dataset = MockROSMAPDataset(
            n_samples=16,
            n_genes=n_genes,
            max_cells=max_cells,
            mode="mixed",
        )
        loader = DataLoader(
            dataset,
            batch_size=4,
            collate_fn=collate_for_hgt_multiregion,
            num_workers=0,
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Simulate 2 epochs
        for epoch in range(2):
            total_loss = 0.0
            for batch in loader:
                optimizer.zero_grad()

                model_input = convert_multiregion_collated_to_model_input(batch)
                output = model(**model_input)

                # Simple MSE loss against cognition target
                loss = torch.nn.functional.mse_loss(
                    output["mean"],
                    model_input["cognition"],
                )
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            # Loss should be finite
            assert not torch.isnan(torch.tensor(total_loss))


class TestEdgeCasesWithFullModel:
    """Test edge cases with the full data-to-model pipeline."""

    def test_batch_size_one(self, model, n_genes, max_cells):
        """Batch size of 1 should work."""
        batch = [create_mock_dataset_sample(n_genes=n_genes, max_cells=max_cells)]
        collated = collate_fn(batch)
        model_input = convert_collated_batch_to_model_input(collated, n_genes)

        output = model(**model_input)

        assert output["mean"].shape == (1, 1)
        assert torch.isfinite(output["mean"]).all()

    def test_empty_ccc_edges(self, model, n_genes, max_cells):
        """Samples with no CCC edges should work.

        # Canonical test for empty CCC edges — see also
        # test_data_to_model.py::TestPipelineEdgeCases::test_pipeline_with_empty_graphs
        # for collate-only coverage.
        """
        sample = create_mock_dataset_sample(n_genes=n_genes, max_cells=max_cells)
        sample["ccc_edge_index"] = torch.zeros(2, 0, dtype=torch.long)
        sample["ccc_edge_type"] = torch.zeros(0, dtype=torch.long)
        sample["ccc_edge_attr"] = torch.zeros(0, 1)

        batch = [sample for _ in range(4)]
        collated = collate_fn(batch)
        model_input = convert_collated_batch_to_model_input(collated, n_genes)

        output = model(**model_input)

        assert torch.isfinite(output["mean"]).all()

    # Sparse cell masks: canonical test in
    # test_full_model_integration.py::TestNumericalStability::test_sparse_cell_masks
    # See also test_data_to_model.py::TestEndToEndPipeline::test_pipeline_with_sparse_data
    # for CellTransformer-only coverage.

    def test_only_one_region_available_in_multi_region_format(self, model, n_genes, max_cells):
        """Multi-region collate with only one region available should work."""
        batch = [
            create_single_region_sample(n_genes=n_genes, max_cells=max_cells)
            for _ in range(4)
        ]
        collated = collate_for_hgt_multiregion(batch)
        model_input = convert_multiregion_collated_to_model_input(collated)

        output = model(**model_input)

        assert torch.isfinite(output["mean"]).all()

    def test_varying_batch_sizes_through_loader(self, model, n_genes, max_cells):
        """Different batch sizes should all work."""
        for batch_size in [1, 2, 4, 8]:
            dataset = MockROSMAPDataset(
                n_samples=batch_size,
                n_genes=n_genes,
                max_cells=max_cells,
                mode="single_region",
            )
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                collate_fn=collate_fn,
                num_workers=0,
            )

            for batch in loader:
                model_input = convert_collated_batch_to_model_input(batch, n_genes)
                output = model(**model_input)

                assert output["mean"].shape == (batch_size, 1)
                assert torch.isfinite(output["mean"]).all()


class TestEndToEndSyntheticPipeline:
    """End-to-end test: synthetic CognitiveResilienceDataset -> collate -> model.

    Verifies the full pipeline from real Dataset objects (not mock samples)
    through collation and model forward pass.
    """

    @pytest.fixture
    def synthetic_dataset(self):
        """Create a small synthetic CognitiveResilienceDataset with real AnnData."""
        import numpy as np
        import pandas as pd
        import anndata
        from src.data.datasets import CognitiveResilienceDataset

        n_cells = 50
        n_genes = 100
        n_subjects = 5

        np.random.seed(42)

        X = np.random.rand(n_cells, n_genes).astype(np.float32)
        obs = pd.DataFrame({
            "ROSMAP_IndividualID": np.repeat(
                [f"synth_{i:03d}" for i in range(n_subjects)],
                n_cells // n_subjects,
            ),
            "supercluster_name": np.random.choice(CELL_TYPE_ORDER, n_cells),
            "BrainRegion": np.random.choice(["PFC", "AG", "MTC"], n_cells),
        })
        var = pd.DataFrame(index=[f"gene_{i}" for i in range(n_genes)])
        adata = anndata.AnnData(X=X, obs=obs, var=var)

        metadata = pd.DataFrame({
            "ROSMAP_IndividualID": [f"synth_{i:03d}" for i in range(n_subjects)],
            "gpath": np.random.rand(n_subjects).astype(np.float32),
            "amylsqrt": np.random.rand(n_subjects).astype(np.float32),
            "tangsqrt": np.random.rand(n_subjects).astype(np.float32),
            "cogn_global": np.random.randn(n_subjects).astype(np.float32),
        })

        subject_ids = [f"synth_{i:03d}" for i in range(n_subjects)]

        return CognitiveResilienceDataset(
            adata=adata,
            metadata=metadata,
            subject_ids=subject_ids,
            max_cells_per_type=20,
        )

    @pytest.fixture
    def synthetic_model(self):
        """Create a small CognitiveResilienceModel matching synthetic data dims."""
        return CognitiveResilienceModel(
            n_genes=100,
            n_cell_types=N_CELL_TYPES,
            d_embed=32,
            d_fused=32,
            d_cond=16,
            n_regions=N_REGIONS,
            n_hgt_layers=1,
            n_hgt_heads=2,
            n_isab_layers=1,
            n_inducing_points=8,
            n_attention_heads=2,
            d_head_hidden=16,
            dropout=0.0,
            use_bayesian_head=False,
        )

    def test_dataset_to_collate_to_model(self, synthetic_dataset, synthetic_model):
        """Full pipeline: Dataset[0:2] -> collate_for_hgt_multiregion -> model forward."""
        # 1. Get samples from the real dataset
        sample_0 = synthetic_dataset[0]
        sample_1 = synthetic_dataset[1]

        # 2. Collate using the multiregion HGT collate function
        collated = collate_for_hgt_multiregion([sample_0, sample_1])

        # 3. Convert collated batch to model input format
        model_input = convert_multiregion_collated_to_model_input(collated)

        # 4. Forward pass through the model
        output = synthetic_model(**model_input)

        # 5. Assertions on output
        assert "mean" in output, "Output must contain 'mean' key"
        assert output["mean"].shape == (2, 1), (
            f"Expected output shape (2, 1), got {output['mean'].shape}"
        )
        assert torch.isfinite(output["mean"]).all(), "Output contains NaN or Inf"
        assert "attention_weights" in output, "Output must contain 'attention_weights'"

    def test_dataset_to_collate_to_bayesian_model(self, synthetic_dataset):
        """Full pipeline with Bayesian head: output should include 'std'."""
        bayesian_model = CognitiveResilienceModel(
            n_genes=100,
            n_cell_types=N_CELL_TYPES,
            d_embed=32,
            d_fused=32,
            d_cond=16,
            n_regions=N_REGIONS,
            n_hgt_layers=1,
            n_hgt_heads=2,
            n_isab_layers=1,
            n_inducing_points=8,
            n_attention_heads=2,
            d_head_hidden=16,
            dropout=0.0,
            use_bayesian_head=True,
        )

        sample_0 = synthetic_dataset[0]
        sample_1 = synthetic_dataset[1]
        collated = collate_for_hgt_multiregion([sample_0, sample_1])
        model_input = convert_multiregion_collated_to_model_input(collated)

        output = bayesian_model(**model_input)

        assert "mean" in output
        assert "std" in output, "Bayesian model output must contain 'std'"
        assert output["mean"].shape == (2, 1)
        assert output["std"].shape == (2, 1)
        assert torch.isfinite(output["mean"]).all()
        assert torch.isfinite(output["std"]).all()
        assert (output["std"] > 0).all(), "Bayesian std must be positive"
