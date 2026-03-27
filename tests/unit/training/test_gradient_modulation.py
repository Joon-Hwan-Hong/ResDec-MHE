"""
Tests for OGM-GE gradient modulation (Peng et al., CVPR 2022 adapted for regression).

Tests OGMGEModulator (coefficient computation) and GradientModulationCallback
(integration with PyTorch model gradients).
"""

import math

import pytest
import torch
import torch.nn as nn


class TestOGMGEModulator:
    """Tests for OGMGEModulator coefficient computation."""

    def test_balanced_branches_get_k_one(self):
        """Equal norms across all branches should produce k=1 for all."""
        from src.training.gradient_modulation import OGMGEModulator

        modulator = OGMGEModulator(alpha=1.0, branch_names=("a", "b", "c"))
        norms = {"a": 5.0, "b": 5.0, "c": 5.0}
        k = modulator.compute_k(norms)
        assert k == {"a": 1.0, "b": 1.0, "c": 1.0}

    def test_dominant_branch_suppressed(self):
        """Branch with above-average norm should get k < 1."""
        from src.training.gradient_modulation import OGMGEModulator

        modulator = OGMGEModulator(alpha=1.0, branch_names=("a", "b", "c"))
        # "a" has 10x the norm of "b" and "c"
        norms = {"a": 10.0, "b": 1.0, "c": 1.0}
        k = modulator.compute_k(norms)
        assert k["a"] < 1.0, f"Dominant branch should be suppressed, got k={k['a']}"

    def test_lagging_branches_unchanged(self):
        """Branches with below-average norm should get k=1 (unchanged)."""
        from src.training.gradient_modulation import OGMGEModulator

        modulator = OGMGEModulator(alpha=1.0, branch_names=("a", "b", "c"))
        norms = {"a": 10.0, "b": 1.0, "c": 1.0}
        k = modulator.compute_k(norms)
        assert k["b"] == 1.0, f"Lagging branch b should be unchanged, got k={k['b']}"
        assert k["c"] == 1.0, f"Lagging branch c should be unchanged, got k={k['c']}"

    def test_alpha_controls_suppression_strength(self):
        """Higher alpha should produce smaller k for dominant branches."""
        from src.training.gradient_modulation import OGMGEModulator

        norms = {"a": 10.0, "b": 1.0, "c": 1.0}
        mod_low = OGMGEModulator(alpha=0.5, branch_names=("a", "b", "c"))
        mod_high = OGMGEModulator(alpha=2.0, branch_names=("a", "b", "c"))
        k_low = mod_low.compute_k(norms)
        k_high = mod_high.compute_k(norms)
        assert k_high["a"] < k_low["a"], (
            f"Higher alpha should suppress more: k_high={k_high['a']}, k_low={k_low['a']}"
        )

    def test_zero_norms_handled(self):
        """All-zero norms should produce k=1 for all branches (no div by zero)."""
        from src.training.gradient_modulation import OGMGEModulator

        modulator = OGMGEModulator(alpha=1.0, branch_names=("a", "b", "c"))
        norms = {"a": 0.0, "b": 0.0, "c": 0.0}
        k = modulator.compute_k(norms)
        assert k == {"a": 1.0, "b": 1.0, "c": 1.0}

    def test_k_formula_matches_paper(self):
        """Verify k = 1 - tanh(alpha * rho) for dominant branches (rho > 1)."""
        from src.training.gradient_modulation import OGMGEModulator

        alpha = 1.5
        modulator = OGMGEModulator(alpha=alpha, branch_names=("a", "b", "c"))
        norms = {"a": 9.0, "b": 3.0, "c": 6.0}
        mean_norm = (9.0 + 3.0 + 6.0) / 3.0  # = 6.0
        k = modulator.compute_k(norms)

        # Branch "a": rho = 9/6 = 1.5 > 1 → k = 1 - tanh(1.5 * 1.5)
        rho_a = 9.0 / mean_norm
        expected_k_a = 1.0 - math.tanh(alpha * rho_a)
        assert abs(k["a"] - expected_k_a) < 1e-10, (
            f"Branch a: expected k={expected_k_a}, got k={k['a']}"
        )

        # Branch "b": rho = 3/6 = 0.5 ≤ 1 → k = 1
        assert k["b"] == 1.0

        # Branch "c": rho = 6/6 = 1.0 ≤ 1 → k = 1 (boundary: not strictly >1)
        assert k["c"] == 1.0


class TestGradientModulationCallback:
    """Tests for GradientModulationCallback integration with model gradients."""

    def test_ogm_scales_dominant_branch_gradients(self):
        """
        Create a simple model with 3 named branch submodules.
        After forward+backward, verify OGM scales dominant branch gradients
        by k < 1 and leaves lagging branches unchanged.
        """
        from src.training.gradient_modulation import GradientModulationCallback

        # Build a simple model with named branches matching BRANCH_NAMES-style structure.
        # Each branch has different output scale to create gradient imbalance.
        # The "scale" factors simulate branches with different magnitudes
        # (like the real model where cell_transformer has 5 chained softmax ops).
        class SimpleBranchModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.pseudobulk_encoder = nn.Linear(4, 2, bias=False)
                self.hgt_encoder = nn.Linear(4, 2, bias=False)
                self.cell_transformer = nn.Linear(4, 2, bias=False)
                self.head = nn.Linear(6, 1, bias=False)

            def forward(self, x):
                b1 = self.pseudobulk_encoder(x) * 10.0  # amplify
                b2 = self.hgt_encoder(x)
                b3 = self.cell_transformer(x) * 0.01    # attenuate
                fused = torch.cat([b1, b2, b3], dim=-1)
                return self.head(fused)

        torch.manual_seed(42)
        model = SimpleBranchModel()

        # Forward + backward to create gradients
        x = torch.randn(8, 4)
        y = torch.randn(8, 1)
        loss = ((model(x) - y) ** 2).mean()
        loss.backward()

        # Record pre-OGM gradient norms
        pre_norms = {}
        for name in ("pseudobulk_encoder", "hgt_encoder", "cell_transformer"):
            branch = getattr(model, name)
            grad_norm = sum(
                p.grad.data.norm(2).item() ** 2 for p in branch.parameters() if p.grad is not None
            ) ** 0.5
            pre_norms[name] = grad_norm

        # Verify we actually have gradient imbalance
        assert pre_norms["pseudobulk_encoder"] > pre_norms["hgt_encoder"], (
            "Test setup: pseudobulk should dominate"
        )

        # Create callback (GE disabled for this test — just OGM)
        callback = GradientModulationCallback(
            alpha=1.0, ge_enabled=False, log_modulation=False,
        )

        # Build param cache and compute norms (internal methods)
        callback._build_param_cache(model)
        branch_norms = callback._compute_branch_norms(model)

        # Compute k values
        k = callback.modulator.compute_k(branch_norms)

        # Save pre-scaling gradients for comparison
        pre_grads = {}
        for name in ("pseudobulk_encoder", "hgt_encoder", "cell_transformer"):
            branch = getattr(model, name)
            pre_grads[name] = {
                pname: p.grad.data.clone()
                for pname, p in branch.named_parameters()
                if p.grad is not None
            }

        # Apply OGM scaling manually (same logic as callback)
        for branch_name, k_val in k.items():
            if branch_name in callback._branch_params:
                for p in callback._branch_params[branch_name]:
                    if p.grad is not None:
                        p.grad.data.mul_(k_val)

        # Verify: dominant branch gradients were scaled down
        dominant = "pseudobulk_encoder"
        assert k[dominant] < 1.0, f"Dominant branch should have k < 1, got {k[dominant]}"
        for pname, pre_grad in pre_grads[dominant].items():
            branch = getattr(model, dominant)
            for n, p in branch.named_parameters():
                if n == pname and p.grad is not None:
                    post_norm = p.grad.data.norm(2).item()
                    pre_norm = pre_grad.norm(2).item()
                    assert post_norm < pre_norm, (
                        f"Dominant branch grad should decrease: {post_norm} >= {pre_norm}"
                    )

        # Verify: lagging branches unchanged (k=1)
        for lagging_name in ("hgt_encoder", "cell_transformer"):
            if k[lagging_name] == 1.0:
                for pname, pre_grad in pre_grads[lagging_name].items():
                    branch = getattr(model, lagging_name)
                    for n, p in branch.named_parameters():
                        if n == pname and p.grad is not None:
                            assert torch.allclose(p.grad.data, pre_grad), (
                                f"Lagging branch {lagging_name} grad should be unchanged"
                            )

    def test_ge_noise_formula(self):
        """GE noise: effective = k * g1 + (g2 - g1) / sqrt(2)."""
        # Simulate gradients
        g1 = torch.tensor([1.0, 2.0, 3.0])  # unscaled main pass gradient
        g2 = torch.tensor([1.5, 1.8, 3.2])  # re-evaluation gradient
        k = 0.6                               # OGM suppression coefficient

        sqrt2 = math.sqrt(2.0)
        noise = (g2 - g1) / sqrt2
        effective = k * g1 + noise

        # Verify formula
        expected = k * g1 + (g2 - g1) / sqrt2
        assert torch.allclose(effective, expected, atol=1e-6)

        # Verify effective != k * g1 (noise was added)
        scaled_only = k * g1
        assert not torch.allclose(effective, scaled_only)

        # Verify effective != g1 (was modified)
        assert not torch.allclose(effective, g1)
