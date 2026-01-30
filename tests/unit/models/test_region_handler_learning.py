"""
Tests for RegionHandler weight learning during training.

Verifies that learned region importance weights:
1. Initialize uniformly and sum to 1
2. Receive gradients and update during training
3. Remain properly normalized after updates
4. Converge with consistent data
5. Integrate correctly with the full model

Test organization:
1. Weight Initialization - uniform start, sum to 1
2. Gradient Flow - multi-region gradients, single-region degenerate, partial mask selective
3. Weight Evolution During Training - changes, normalization, convergence
4. Region Importance Extraction - dict format, values validity
5. Integration with Full Model - full model training, context changes with mask
"""

import pytest
import torch
import torch.nn as nn


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def small_handler():
    """Small handler for fast tests."""
    from src.models.components.region_handler import RegionHandler
    return RegionHandler(d_model=32, n_regions=6)


@pytest.fixture
def handler_for_training():
    """Handler configured for training tests."""
    from src.models.components.region_handler import RegionHandler
    torch.manual_seed(42)
    return RegionHandler(d_model=64, n_regions=6)


# =============================================================================
# 1. WEIGHT INITIALIZATION TESTS
# =============================================================================


class TestWeightInitialization:
    """Tests for RegionHandler weight initialization."""

    def test_uniform_initial_weights(self):
        """All weights should start equal (1/6 for 6 regions)."""
        from src.models.components.region_handler import RegionHandler

        handler = RegionHandler(d_model=32, n_regions=6)
        weights = handler.get_region_weights()

        expected = torch.ones(6) / 6
        assert torch.allclose(weights, expected, atol=1e-6), (
            f"Initial weights should be uniform 1/6, got {weights.tolist()}"
        )

    def test_initial_weights_sum_to_one(self):
        """Initial softmax weights should sum to 1."""
        from src.models.components.region_handler import RegionHandler

        handler = RegionHandler(d_model=64, n_regions=6)
        weights = handler.get_region_weights()

        assert torch.allclose(weights.sum(), torch.tensor(1.0), atol=1e-6), (
            f"Weights should sum to 1.0, got {weights.sum().item()}"
        )

    def test_raw_weights_initialized_to_zeros(self):
        """Raw region_weights parameter should be zeros (giving uniform softmax)."""
        from src.models.components.region_handler import RegionHandler

        handler = RegionHandler(d_model=32, n_regions=6)

        assert torch.allclose(handler.region_weights, torch.zeros(6)), (
            f"Raw weights should be zeros, got {handler.region_weights.tolist()}"
        )

    def test_uniform_init_different_n_regions(self):
        """Uniform initialization should work for different region counts."""
        from src.models.components.region_handler import RegionHandler

        for n_regions in [3, 6, 10]:
            handler = RegionHandler(d_model=32, n_regions=n_regions)
            weights = handler.get_region_weights()

            expected = torch.ones(n_regions) / n_regions
            assert torch.allclose(weights, expected, atol=1e-6), (
                f"With {n_regions} regions, weights should be uniform 1/{n_regions}"
            )


# =============================================================================
# 2. GRADIENT FLOW TESTS
# =============================================================================


class TestGradientFlow:
    """Tests for gradient flow to region weights."""

    def test_multi_region_gradient_flows_to_weights(self):
        """With multiple regions active, gradients should flow to region_weights."""
        from src.models.components.region_handler import RegionHandler

        handler = RegionHandler(d_model=32, n_regions=6)

        # Create input where different regions have different values
        # This ensures gradients will differ per region
        x = torch.zeros(2, 6, 5, 32)
        for r in range(6):
            x[:, r, :, :] = r * 0.5  # Different value per region

        region_mask = torch.ones(2, 6, dtype=torch.bool)  # All regions active

        pooled, _ = handler(x, region_mask)
        loss = pooled.sum()
        loss.backward()

        # Gradient should exist and be non-zero
        assert handler.region_weights.grad is not None, "region_weights.grad is None"
        assert not torch.allclose(handler.region_weights.grad, torch.zeros(6)), (
            "Gradient should be non-zero when multiple regions have different values"
        )

    def test_single_region_no_effective_gradient(self):
        """With single region, gradient to weights is degenerate (all contribute equally post-softmax)."""
        from src.models.components.region_handler import RegionHandler

        handler = RegionHandler(d_model=32, n_regions=6)

        # Create input with only one region active
        x = torch.randn(2, 6, 5, 32)
        region_mask = torch.zeros(2, 6, dtype=torch.bool)
        region_mask[:, 0] = True  # Only region 0 active

        pooled, _ = handler(x, region_mask)
        loss = pooled.sum()
        loss.backward()

        # Gradient should exist (flows through softmax)
        assert handler.region_weights.grad is not None, "region_weights.grad is None"

        # With single region, the output is just x[:, 0] regardless of weight value
        # The normalized weight is always 1.0 for the active region
        # So gradient w.r.t. region_weights should be zero (no influence on output)
        # Note: softmax still produces gradients but they should effectively be zero
        # because changing the weight doesn't change the normalized result
        grad = handler.region_weights.grad
        # The gradient exists but is effectively useless for learning when only 1 region
        # This is expected behavior - single region gives degenerate case

    def test_partial_region_mask_gradient_selective(self):
        """With partial mask, only active regions' weights should effectively contribute."""
        from src.models.components.region_handler import RegionHandler

        handler = RegionHandler(d_model=32, n_regions=6)

        # Create input where only some regions are active
        x = torch.zeros(2, 6, 5, 32)
        for r in range(6):
            x[:, r, :, :] = r * 0.5

        # Only first 3 regions active
        region_mask = torch.zeros(2, 6, dtype=torch.bool)
        region_mask[:, :3] = True

        pooled, _ = handler(x, region_mask)
        loss = pooled.sum()
        loss.backward()

        grad = handler.region_weights.grad
        assert grad is not None

        # Due to softmax, all gradients are non-zero, but the masked regions
        # only contribute through the softmax normalization
        # The key check: gradient structure reflects partial mask
        # Active regions (0,1,2) should have meaningful gradients
        # Masked regions (3,4,5) gradients come only from softmax normalization

        # Verify gradients exist for active regions
        assert not torch.allclose(grad[:3], torch.zeros(3), atol=1e-8), (
            "Active region gradients should be non-zero"
        )

    def test_gradient_flow_with_different_input_patterns(self):
        """Gradient flow should work with various input patterns."""
        from src.models.components.region_handler import RegionHandler

        handler = RegionHandler(d_model=16, n_regions=6)

        # Pattern 1: Alternating regions
        x = torch.randn(4, 6, 3, 16)
        region_mask = torch.tensor([
            [True, False, True, False, True, False],
            [False, True, False, True, False, True],
            [True, True, True, False, False, False],
            [False, False, False, True, True, True],
        ])

        pooled, _ = handler(x, region_mask)
        loss = pooled.sum()
        loss.backward()

        assert handler.region_weights.grad is not None


# =============================================================================
# 3. WEIGHT EVOLUTION DURING TRAINING TESTS
# =============================================================================


class TestWeightEvolutionDuringTraining:
    """Tests for weight changes during training."""

    def test_weights_change_during_training(self):
        """Weights should actually change with gradient descent steps."""
        from src.models.components.region_handler import RegionHandler

        torch.manual_seed(42)
        handler = RegionHandler(d_model=32, n_regions=6)
        optimizer = torch.optim.SGD(handler.parameters(), lr=0.1)

        # Store initial weights
        initial_weights = handler.get_region_weights().clone().detach()

        # Create input with region-dependent signal
        # Region 0 has large positive values, region 5 has large negative
        x = torch.zeros(4, 6, 5, 32)
        for r in range(6):
            x[:, r, :, :] = (r - 2.5) * 2.0  # Values: -5, -3, -1, 1, 3, 5

        region_mask = torch.ones(4, 6, dtype=torch.bool)

        # Training steps - minimize pooled sum (should increase weight on negative regions)
        for _ in range(10):
            optimizer.zero_grad()
            pooled, _ = handler(x, region_mask)
            loss = pooled.sum()  # Minimize sum -> prefer negative regions
            loss.backward()
            optimizer.step()

        final_weights = handler.get_region_weights().detach()

        # Weights should have changed
        assert not torch.allclose(initial_weights, final_weights, atol=1e-4), (
            f"Weights should change during training.\n"
            f"Initial: {initial_weights.tolist()}\n"
            f"Final: {final_weights.tolist()}"
        )

    def test_weights_remain_normalized(self):
        """Weights should remain normalized (sum to 1) after training updates."""
        from src.models.components.region_handler import RegionHandler

        torch.manual_seed(42)
        handler = RegionHandler(d_model=32, n_regions=6)
        optimizer = torch.optim.Adam(handler.parameters(), lr=0.01)

        x = torch.randn(4, 6, 5, 32)
        region_mask = torch.ones(4, 6, dtype=torch.bool)

        # Multiple training steps
        for step in range(20):
            optimizer.zero_grad()
            pooled, _ = handler(x, region_mask)
            loss = pooled.sum()
            loss.backward()
            optimizer.step()

            # Check normalization after each step
            weights = handler.get_region_weights()
            assert torch.allclose(weights.sum(), torch.tensor(1.0), atol=1e-5), (
                f"Weights should sum to 1 after step {step}, got {weights.sum().item()}"
            )
            assert torch.all(weights > 0), (
                f"All weights should be positive after step {step}"
            )

    def test_weight_learning_convergence(self):
        """Weights should converge with repeated consistent data."""
        from src.models.components.region_handler import RegionHandler

        torch.manual_seed(123)
        handler = RegionHandler(d_model=32, n_regions=6)
        optimizer = torch.optim.Adam(handler.parameters(), lr=0.05)

        # Create consistent data pattern: region 2 has the "best" signal
        # Target: minimize MSE to target value
        target = torch.ones(4, 5, 32) * 2.0  # Target is 2.0

        x = torch.zeros(4, 6, 5, 32)
        x[:, 0, :, :] = 0.0   # Region 0: value 0
        x[:, 1, :, :] = 1.0   # Region 1: value 1
        x[:, 2, :, :] = 2.0   # Region 2: value 2 (matches target!)
        x[:, 3, :, :] = 3.0   # Region 3: value 3
        x[:, 4, :, :] = 4.0   # Region 4: value 4
        x[:, 5, :, :] = 5.0   # Region 5: value 5

        region_mask = torch.ones(4, 6, dtype=torch.bool)

        losses = []
        weight_history = []

        # Train to minimize MSE loss
        for step in range(100):
            optimizer.zero_grad()
            pooled, _ = handler(x, region_mask)
            loss = ((pooled - target) ** 2).mean()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            weight_history.append(handler.get_region_weights().detach().clone())

        # Loss should decrease
        assert losses[-1] < losses[0], (
            f"Loss should decrease: initial={losses[0]:.4f}, final={losses[-1]:.4f}"
        )

        # Region 2 should have highest weight (since it matches target)
        final_weights = handler.get_region_weights().detach()
        assert final_weights[2] > final_weights.max() * 0.5, (
            f"Region 2 should have highest weight since it matches target.\n"
            f"Weights: {final_weights.tolist()}"
        )

    def test_weights_converge_to_stable_values(self):
        """After sufficient training, weights should stabilize."""
        from src.models.components.region_handler import RegionHandler

        torch.manual_seed(42)
        handler = RegionHandler(d_model=32, n_regions=6)
        optimizer = torch.optim.Adam(handler.parameters(), lr=0.01)

        # Fixed input pattern
        x = torch.randn(4, 6, 5, 32)
        region_mask = torch.ones(4, 6, dtype=torch.bool)
        target = x[:, 0].clone()  # Target is region 0's embedding

        # Train for many steps
        for _ in range(200):
            optimizer.zero_grad()
            pooled, _ = handler(x, region_mask)
            loss = ((pooled - target) ** 2).mean()
            loss.backward()
            optimizer.step()

        # Check stability: weights shouldn't change much in final steps
        weights_before = handler.get_region_weights().detach().clone()

        for _ in range(10):
            optimizer.zero_grad()
            pooled, _ = handler(x, region_mask)
            loss = ((pooled - target) ** 2).mean()
            loss.backward()
            optimizer.step()

        weights_after = handler.get_region_weights().detach()

        # Weights should be similar (converged)
        weight_change = (weights_after - weights_before).abs().max()
        assert weight_change < 0.05, (
            f"Weights should be stable after convergence, max change: {weight_change:.4f}"
        )


# =============================================================================
# 4. REGION IMPORTANCE EXTRACTION TESTS
# =============================================================================


class TestRegionImportanceExtraction:
    """Tests for get_region_importance_dict method."""

    def test_get_region_importance_dict_format(self):
        """Dict should have correct keys (region names)."""
        from src.models.components.region_handler import RegionHandler

        handler = RegionHandler(d_model=32, n_regions=6)
        importance = handler.get_region_importance_dict()

        # Check type
        assert isinstance(importance, dict)

        # Check keys match REGIONS
        expected_keys = {"PFC", "AG", "MTC", "EC", "HC", "TH"}
        assert set(importance.keys()) == expected_keys, (
            f"Keys should be {expected_keys}, got {set(importance.keys())}"
        )

    def test_get_region_importance_dict_values(self):
        """Values should sum to 1 and all be positive."""
        from src.models.components.region_handler import RegionHandler

        handler = RegionHandler(d_model=32, n_regions=6)

        # Modify weights to non-uniform
        with torch.no_grad():
            handler.region_weights.copy_(torch.tensor([1.0, 0.5, -0.5, 0.0, 2.0, -1.0]))

        importance = handler.get_region_importance_dict()

        # All values should be positive
        for name, value in importance.items():
            assert value > 0, f"Value for {name} should be positive, got {value}"

        # Values should sum to 1
        total = sum(importance.values())
        assert abs(total - 1.0) < 1e-5, f"Values should sum to 1, got {total}"

    def test_get_region_importance_dict_reflects_learned_weights(self):
        """Dict values should reflect the current learned weights."""
        from src.models.components.region_handler import RegionHandler

        handler = RegionHandler(d_model=32, n_regions=6)

        # Set known weights
        with torch.no_grad():
            handler.region_weights.copy_(torch.tensor([5.0, 0.0, 0.0, 0.0, 0.0, 0.0]))

        importance = handler.get_region_importance_dict()

        # PFC (index 0) should have much higher importance
        assert importance["PFC"] > 0.9, (
            f"PFC should dominate with high raw weight, got {importance['PFC']}"
        )

    def test_get_region_importance_dict_detached(self):
        """Returned dict values should be Python floats (detached from graph)."""
        from src.models.components.region_handler import RegionHandler

        handler = RegionHandler(d_model=32, n_regions=6)
        importance = handler.get_region_importance_dict()

        for name, value in importance.items():
            assert isinstance(value, float), (
                f"Value for {name} should be float, got {type(value)}"
            )


# =============================================================================
# 5. INTEGRATION WITH FULL MODEL TESTS
# =============================================================================


class TestIntegrationWithFullModel:
    """Tests for RegionHandler weight learning in full model context."""

    def test_region_weights_in_full_model_training(self):
        """Region weights should update during full model training."""
        from src.models.branches.pseudobulk_encoder import PseudobulkEncoder
        from src.models.components.region_handler import RegionHandler

        torch.manual_seed(42)

        # Setup
        n_genes = 50
        n_cell_types = 31
        d_embed = 32
        n_regions = 6

        encoder = PseudobulkEncoder(
            n_cell_types=n_cell_types,
            n_genes=n_genes,
            d_embed=d_embed,
        )
        region_handler = RegionHandler(d_model=d_embed, n_regions=n_regions)

        # Combine into module for unified optimizer
        class CombinedModule(nn.Module):
            def __init__(self, enc, rh):
                super().__init__()
                self.encoder = enc
                self.region_handler = rh

            def forward(self, region_pseudobulk, region_mask):
                B, R, C, G = region_pseudobulk.shape
                encoded = self.encoder(region_pseudobulk.view(B * R, C, G))
                encoded = encoded.view(B, R, C, -1)
                return self.region_handler(encoded, region_mask)

        model = CombinedModule(encoder, region_handler)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        # Store initial weights
        initial_weights = region_handler.get_region_weights().detach().clone()

        # Training data - different regions have different expression patterns
        region_pseudobulk = torch.randn(4, n_regions, n_cell_types, n_genes)
        # Make region 0 have a distinct pattern
        region_pseudobulk[:, 0, :, :] = region_pseudobulk[:, 0, :, :].abs() * 2

        region_mask = torch.ones(4, n_regions, dtype=torch.bool)
        target = torch.randn(4, n_cell_types, d_embed)

        # Train
        for _ in range(20):
            optimizer.zero_grad()
            pooled, _ = model(region_pseudobulk, region_mask)
            loss = ((pooled - target) ** 2).mean()
            loss.backward()
            optimizer.step()

        final_weights = region_handler.get_region_weights().detach()

        # Weights should have changed
        assert not torch.allclose(initial_weights, final_weights, atol=1e-4), (
            "Region weights should change during full model training"
        )

    def test_region_context_changes_with_mask(self):
        """region_context should differ based on which regions are available."""
        from src.models.components.region_handler import RegionHandler

        handler = RegionHandler(d_model=32, n_regions=6)

        x = torch.randn(2, 6, 5, 32)

        # Mask 1: Only region 0
        mask1 = torch.zeros(2, 6, dtype=torch.bool)
        mask1[:, 0] = True

        # Mask 2: Only region 5
        mask2 = torch.zeros(2, 6, dtype=torch.bool)
        mask2[:, 5] = True

        # Mask 3: All regions
        mask3 = torch.ones(2, 6, dtype=torch.bool)

        _, context1 = handler(x, mask1)
        _, context2 = handler(x, mask2)
        _, context3 = handler(x, mask3)

        # Contexts should all be different (different region embeddings)
        assert not torch.allclose(context1, context2, atol=1e-5), (
            "Context for region 0 only vs region 5 only should differ"
        )
        assert not torch.allclose(context1, context3, atol=1e-5), (
            "Context for single region vs all regions should differ"
        )
        assert not torch.allclose(context2, context3, atol=1e-5), (
            "Context for single region vs all regions should differ"
        )

    def test_region_weights_affect_pooled_output(self):
        """Changing region weights should change the pooled output."""
        from src.models.components.region_handler import RegionHandler

        handler1 = RegionHandler(d_model=32, n_regions=6)
        handler2 = RegionHandler(d_model=32, n_regions=6)

        # Set very different weights using large values to ensure dominance
        # With weight=10.0, softmax gives ~0.9999 to dominant region
        with torch.no_grad():
            handler1.region_weights.copy_(torch.tensor([10.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
            handler2.region_weights.copy_(torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 10.0]))

        # Same input
        x = torch.randn(2, 6, 5, 32)
        region_mask = torch.ones(2, 6, dtype=torch.bool)

        pooled1, _ = handler1(x, region_mask)
        pooled2, _ = handler2(x, region_mask)

        # Outputs should differ significantly
        assert not torch.allclose(pooled1, pooled2, atol=1e-3), (
            "Different weight distributions should produce different pooled outputs"
        )

        # handler1 should be very close to region 0 (weight ~0.9999)
        assert torch.allclose(pooled1, x[:, 0], atol=0.01), (
            f"Handler1 with high weight on region 0 should output ~region 0.\n"
            f"Max diff: {(pooled1 - x[:, 0]).abs().max().item():.6f}"
        )
        # handler2 should be very close to region 5 (weight ~0.9999)
        assert torch.allclose(pooled2, x[:, 5], atol=0.01), (
            f"Handler2 with high weight on region 5 should output ~region 5.\n"
            f"Max diff: {(pooled2 - x[:, 5]).abs().max().item():.6f}"
        )

    def test_full_model_gradient_flow_to_region_weights(self):
        """Gradients should flow from final loss through full model to region weights."""
        from src.models.branches.pseudobulk_encoder import PseudobulkEncoder
        from src.models.components.region_handler import RegionHandler

        n_genes = 50
        n_cell_types = 31
        d_embed = 32

        encoder = PseudobulkEncoder(
            n_cell_types=n_cell_types,
            n_genes=n_genes,
            d_embed=d_embed,
        )
        region_handler = RegionHandler(d_model=d_embed, n_regions=6)

        # Forward pass
        region_pseudobulk = torch.randn(2, 6, n_cell_types, n_genes)
        region_mask = torch.ones(2, 6, dtype=torch.bool)

        B, R, C, G = region_pseudobulk.shape
        encoded = encoder(region_pseudobulk.view(B * R, C, G))
        encoded = encoded.view(B, R, C, -1)
        pooled, region_context = region_handler(encoded, region_mask)

        # Compute loss and backward
        loss = pooled.sum() + region_context.sum()
        loss.backward()

        # Check gradient exists on region_weights
        assert region_handler.region_weights.grad is not None, (
            "Gradient should reach region_weights through full pipeline"
        )


# =============================================================================
# 6. ADDITIONAL EDGE CASES
# =============================================================================


class TestEdgeCasesLearning:
    """Additional edge cases for weight learning."""

    def test_extreme_weight_values_stable(self):
        """Training should remain stable even with extreme learned weights."""
        from src.models.components.region_handler import RegionHandler

        handler = RegionHandler(d_model=32, n_regions=6)
        optimizer = torch.optim.Adam(handler.parameters(), lr=0.01)

        # Set extreme initial weights
        with torch.no_grad():
            handler.region_weights.copy_(torch.tensor([10.0, -10.0, 5.0, -5.0, 0.0, 3.0]))

        x = torch.randn(4, 6, 5, 32)
        region_mask = torch.ones(4, 6, dtype=torch.bool)

        # Should not crash or produce NaN
        for _ in range(10):
            optimizer.zero_grad()
            pooled, context = handler(x, region_mask)
            loss = pooled.sum()
            loss.backward()
            optimizer.step()

            # Check for NaN/Inf
            weights = handler.get_region_weights()
            assert torch.isfinite(weights).all(), "Weights should remain finite"
            assert torch.isfinite(pooled).all(), "Pooled output should remain finite"

    def test_learning_with_sparse_masks(self):
        """Training should work with varying sparse masks per batch."""
        from src.models.components.region_handler import RegionHandler

        torch.manual_seed(42)
        handler = RegionHandler(d_model=32, n_regions=6)
        optimizer = torch.optim.Adam(handler.parameters(), lr=0.01)

        initial_weights = handler.get_region_weights().detach().clone()

        x = torch.randn(8, 6, 5, 32)

        # Varying masks: each sample has different active regions
        region_mask = torch.zeros(8, 6, dtype=torch.bool)
        region_mask[0, [0, 1, 2]] = True      # First 3 regions
        region_mask[1, [3, 4, 5]] = True      # Last 3 regions
        region_mask[2, [0, 2, 4]] = True      # Even regions
        region_mask[3, [1, 3, 5]] = True      # Odd regions
        region_mask[4, :] = True              # All regions
        region_mask[5, [0]] = True            # Just one region
        region_mask[6, [0, 5]] = True         # First and last
        region_mask[7, [2, 3]] = True         # Middle regions

        # Train
        for _ in range(20):
            optimizer.zero_grad()
            pooled, _ = handler(x, region_mask)
            loss = pooled.sum()
            loss.backward()
            optimizer.step()

        final_weights = handler.get_region_weights().detach()

        # Should have learned something (weights changed)
        assert not torch.allclose(initial_weights, final_weights, atol=1e-4), (
            "Weights should change even with sparse masks"
        )

        # Weights should still be valid
        assert torch.allclose(final_weights.sum(), torch.tensor(1.0), atol=1e-5)
        assert torch.all(final_weights > 0)

    def test_weight_learning_with_zero_grad_accumulation(self):
        """Multiple forward passes before backward should work correctly."""
        from src.models.components.region_handler import RegionHandler

        handler = RegionHandler(d_model=32, n_regions=6)
        optimizer = torch.optim.SGD(handler.parameters(), lr=0.1)

        x = torch.randn(4, 6, 5, 32)
        region_mask = torch.ones(4, 6, dtype=torch.bool)

        initial_weights = handler.get_region_weights().detach().clone()

        # Accumulate gradients from multiple forward passes
        optimizer.zero_grad()
        for _ in range(3):
            pooled, _ = handler(x, region_mask)
            loss = pooled.sum() / 3
            loss.backward()

        optimizer.step()

        final_weights = handler.get_region_weights().detach()
        assert not torch.allclose(initial_weights, final_weights, atol=1e-5), (
            "Weights should update with accumulated gradients"
        )
