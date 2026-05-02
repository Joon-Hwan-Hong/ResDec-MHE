"""Shared per-branch parameter cache and gradient-norm helpers.

Used by both ``GradientNormLogger`` (in ``callbacks.py``) and
``GradientModulationCallback`` (in ``gradient_modulation.py``). Extracted
to a single source of truth so a CT-branch rename or added prefix
updates both call sites in lockstep.

The "intentionally duplicated to avoid coupling these two independent
callbacks" rationale (previously in gradient_modulation.py L112) is
incorrect: sharing a utility function does not create logical coupling
between the callbacks; it removes a parallel-edit hazard.
"""
from __future__ import annotations

import torch


# Maps branch name → list of param name substrings that belong to that branch.
# hgt_gene_gate and hgt_input_proj live at the model level but are logically
# part of the HGT encoder branch.
BRANCH_PREFIXES: dict[str, tuple[str, ...]] = {
    "hgt_encoder": ("hgt_encoder", "hgt_gene_gate", "hgt_input_proj"),
    "cell_transformer": ("cell_transformer",),
}


def build_branch_param_cache(
    model: torch.nn.Module,
    branch_names: tuple[str, ...],
    branch_prefixes: dict[str, tuple[str, ...]] | None = None,
) -> dict[str, list[torch.nn.Parameter]]:
    """Build a parameter-to-branch mapping by name-substring match.

    Args:
        model: The model whose ``named_parameters`` are categorised.
        branch_names: Tuple of canonical branch names (keys of the result).
        branch_prefixes: Optional override of the prefix table. Keys are
            branch names; values are tuples of substrings to match against
            parameter names. Branches not in this map default to a single
            prefix matching the branch name itself.

    Returns:
        Dict mapping each branch name to the list of parameters whose name
        contains any of the branch's prefixes. Each parameter is assigned
        to at most one branch (first match wins, in iteration order of
        ``branch_names``).
    """
    prefixes = branch_prefixes if branch_prefixes is not None else BRANCH_PREFIXES
    result: dict[str, list[torch.nn.Parameter]] = {name: [] for name in branch_names}
    for param_name, param in model.named_parameters():
        for branch_name in branch_names:
            keys = prefixes.get(branch_name, (branch_name,))
            if any(key in param_name for key in keys):
                result[branch_name].append(param)
                break
    return result


def compute_branch_norms(
    branch_params: dict[str, list[torch.nn.Parameter]],
) -> dict[str, float]:
    """Compute L2 gradient norm per branch from the param cache.

    Under bf16-mixed and 32-true, ``p.grad`` contains unscaled gradients
    (no GradScaler). Under 16-mixed (float16), Lightning uses GradScaler
    and ``p.grad`` here contains SCALED gradients — logged norms would
    include the scale factor.

    Args:
        branch_params: Output of :func:`build_branch_param_cache`.

    Returns:
        Dict mapping each branch name to its L2 gradient norm. Branches
        with no parameters or no grad-having parameters return 0.0.
    """
    branch_norms: dict[str, float] = {}
    for branch_name, params in branch_params.items():
        grad_params = [p for p in params if p.grad is not None]
        if grad_params:
            # Stack per-param norms and reduce in 2 ops instead of N.
            grad_norms = torch.stack(
                [p.grad.data.norm(2) for p in grad_params]
            )
            branch_norms[branch_name] = grad_norms.norm(2).item()
        else:
            branch_norms[branch_name] = 0.0
    return branch_norms
