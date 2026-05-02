"""Lightning training utilities shared across module classes.

These helpers were extracted to remove repeated try/except patterns and
near-duplicate scheduler builds across ``lightning_module.py``,
``resdec_lightning_module.py``, and elsewhere.
"""
from __future__ import annotations

from typing import Any

import torch
import lightning.pytorch as pl

from src.data.constants import N_REGIONS


def _resolve_trainer(arg: pl.LightningModule | pl.Trainer | None) -> pl.Trainer | None:
    """Resolve a Trainer from either a Trainer or a LightningModule.

    LightningModule.trainer is a property that raises RuntimeError when
    not attached, so we cannot simply pass ``self.trainer`` from a not-
    yet-attached module. Pass the module itself and we resolve safely.
    """
    if arg is None:
        return None
    if isinstance(arg, pl.LightningModule):
        try:
            return arg.trainer  # type: ignore[return-value]
        except RuntimeError:
            return None
    return arg


def world_size_or_one(arg: Any) -> int:
    """Return ``trainer.world_size`` if attached, else 1.

    Lightning raises ``RuntimeError`` when ``self.trainer`` is accessed
    on a module that has not been attached to a Trainer (e.g., during
    unit tests). This wrapper centralises the try/except pattern that
    appeared at multiple call sites in ``lightning_module.py``. Accepts
    either a Trainer instance or a LightningModule; for the latter, the
    Trainer is resolved via ``self.trainer`` with the RuntimeError
    suppressed.
    """
    trainer = _resolve_trainer(arg)
    if trainer is None:
        return 1
    try:
        return int(trainer.world_size)
    except RuntimeError:
        return 1


def is_global_zero_or_true(arg: Any) -> bool:
    """Return ``trainer.is_global_zero`` if attached, else True.

    Mirrors :func:`world_size_or_one` for the rank-zero check; accepts
    either a Trainer or a LightningModule (with the same RuntimeError
    handling for the not-yet-attached case).
    """
    trainer = _resolve_trainer(arg)
    if trainer is None:
        return True
    try:
        return bool(trainer.is_global_zero)
    except RuntimeError:
        return True


def make_prototype_batch(
    model_cfg: Any,
    device: torch.device | str,
    *,
    include_cell_type_mask: bool = True,
    include_ccc: bool = True,
) -> dict[str, torch.Tensor]:
    """Build a minimal dummy batch for guide prototyping / dummy forward.

    Used by both ``CognitiveResilienceLightningModule._prototype_guide_if_needed``
    and ``Predictor.from_checkpoint`` so the prototype shapes stay in sync.

    Args:
        model_cfg: Model config (omegaconf node) exposing ``n_cell_types``,
            ``n_genes``, and optionally ``pathology_attention.n_pathology_features``.
        device: Target device for all returned tensors.
        include_cell_type_mask: Whether to include the boolean
            ``cell_type_mask`` (training/SVI path needs it; the inference
            prototype omits it).
        include_ccc: Whether to include zero-edge CCC tensors (training
            path needs them; inference prototype omits them).

    Returns:
        Dict suitable for passing as kwargs to ``model(...)`` / ``guide(...)``.
    """
    n_ct = int(model_cfg.n_cell_types)
    n_genes = int(model_cfg.n_genes)
    n_pathology = int(
        model_cfg.get("pathology_attention", {}).get("n_pathology_features", 3)
    )

    batch: dict[str, torch.Tensor] = {
        "region_pseudobulk": torch.zeros(
            1, N_REGIONS, n_ct, n_genes, device=device,
        ),
        "region_mask": torch.ones(
            1, N_REGIONS, dtype=torch.bool, device=device,
        ),
        "cell_data": torch.zeros(0, n_genes, device=device),
        "cell_offsets": torch.zeros(
            1, n_ct + 1, dtype=torch.long, device=device,
        ),
        "pathology": torch.zeros(1, n_pathology, device=device),
        "cognition": torch.zeros(1, 1, device=device),
    }
    if include_cell_type_mask:
        batch["cell_type_mask"] = torch.ones(
            1, n_ct, dtype=torch.bool, device=device,
        )
    if include_ccc:
        batch["ccc_edge_index"] = torch.zeros(
            2, 0, dtype=torch.long, device=device,
        )
        batch["ccc_edge_type"] = torch.zeros(
            0, dtype=torch.long, device=device,
        )
        batch["ccc_edge_attr"] = torch.zeros(0, 1, device=device)
    return batch


def build_cosine_warmup_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    max_epochs: int,
    warmup_epochs: int,
    eta_min: float,
    t_max_override: int | None = None,
    warmup_start_factor: float = 0.01,
) -> torch.optim.lr_scheduler.LRScheduler:
    """Build a linear-warmup-then-cosine-anneal LR scheduler.

    Args:
        optimizer: The optimizer whose LR to schedule.
        max_epochs: Total training epochs (used as the cosine cycle
            length when ``t_max_override`` is ``None``).
        warmup_epochs: Number of linear warmup epochs (0 disables warmup).
        eta_min: Final LR floor for cosine annealing.
        t_max_override: Optional explicit ``T_max`` override (epochs of
            cosine descent). Useful when keeping a long-T_max calibration
            while training for fewer epochs.
        warmup_start_factor: ``LinearLR.start_factor`` (default 0.01).

    Returns:
        Either a ``CosineAnnealingLR`` (when ``warmup_epochs == 0``) or
        a ``SequentialLR`` chaining ``LinearLR`` then ``CosineAnnealingLR``.

    Raises:
        ValueError: If ``warmup_epochs >= max_epochs`` and
            ``t_max_override`` is not provided (the cosine cycle length
            would be non-positive).
    """
    if t_max_override is not None:
        t_max = int(t_max_override)
    else:
        t_max = int(max_epochs) - int(warmup_epochs)
    if t_max <= 0:
        raise ValueError(
            f"warmup_epochs ({warmup_epochs}) must be less than "
            f"max_epochs ({max_epochs}) for cosine scheduler"
        )

    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=t_max, eta_min=float(eta_min),
    )

    if warmup_epochs > 0:
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=float(warmup_start_factor),
            end_factor=1.0,
            total_iters=int(warmup_epochs),
        )
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[int(warmup_epochs)],
        )
    return cosine
