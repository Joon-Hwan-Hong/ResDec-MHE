import pytest
import torch
from src.models.resdec_head.npt_stage import NPTStage


def test_npt_stage_shapes():
    stage = NPTStage(d_subject=64, n_heads=4, n_hc_streams=4, lambda_init=0.8)
    z_cond = torch.randn(8, 64)  # 8 subjects in batch
    latent, scalar = stage(z_cond)
    assert latent.shape == (8, 64)
    assert scalar.shape == (8,)


def test_npt_stage_gradient_flow():
    stage = NPTStage(d_subject=32, n_heads=4)
    z_cond = torch.randn(6, 32, requires_grad=True)
    _, scalar = stage(z_cond)
    scalar.sum().backward()
    assert z_cond.grad is not None


def test_npt_stage_attends_across_subjects():
    """Changing ONE subject's input should change OTHER subjects' outputs
    (because NPT attends across the batch axis).
    """
    stage = NPTStage(d_subject=16, n_heads=4)
    stage.eval()
    z1 = torch.randn(4, 16)
    z2 = z1.clone()
    z2[0] = torch.randn(16)  # perturb first subject only
    latent1, _ = stage(z1)
    latent2, _ = stage(z2)
    # Subjects 1-3 should have different latents because NPT attention
    # over the batch axis lets subject-0's change affect them.
    diff_other_subjects = (latent1[1:] - latent2[1:]).abs().max()
    assert diff_other_subjects > 1e-6, "NPT did not attend across subjects"
