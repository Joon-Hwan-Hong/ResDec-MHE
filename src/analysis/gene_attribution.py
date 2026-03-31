"""
Per-subject, per-cell-type gene attribution using Integrated Gradients.

Computes how much each gene in each cell type contributes to the predicted
cognitive score for each subject. Uses Captum's IntegratedGradients with
the pseudobulk expression as the attributed input and a zero baseline.

For Bayesian models: uses forward_encoder_only (branches + fusion + attention)
then applies the Bayesian head's median parameters via F.linear to maintain
autograd compatibility. This gives the same prediction as the trained model's
guide.median() inference path.

Output: [n_subjects, n_cell_types, n_genes] attribution matrix.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from captum.attr import IntegratedGradients
from tqdm import tqdm

logger = logging.getLogger(__name__)


class _PseudobulkForwardWrapper(torch.nn.Module):
    """Wrapper exposing pseudobulk → scalar prediction for Captum.

    Uses forward_encoder_only (no Pyro tracing) then applies the Bayesian
    head layers manually via F.linear with guide median weights. This
    maintains autograd compatibility while using the exact same weights
    the trained Bayesian model uses during inference.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        head_weights: dict[str, torch.Tensor],
    ):
        super().__init__()
        self.model = model
        self.head_weights = head_weights  # guide median + non-Bayesian head params
        self._fixed_kwargs: dict[str, Any] = {}

    def set_fixed_inputs(self, batch: dict[str, Any]) -> None:
        """Set all non-pseudobulk inputs that stay fixed during attribution."""
        self._fixed_kwargs = {
            "region_pseudobulk": batch.get("region_pseudobulk"),
            "region_mask": batch.get("region_mask"),
            "ccc_edge_index": batch.get("ccc_edge_index"),
            "ccc_edge_type": batch.get("ccc_edge_type"),
            "ccc_edge_attr": batch.get("ccc_edge_attr"),
            "cell_type_mask": batch.get("cell_type_mask"),
            "pathology": batch.get("pathology"),
            "cell_data": batch.get("cell_data"),
            "cell_offsets": batch.get("cell_offsets"),
        }

    def forward(self, pseudobulk: torch.Tensor) -> torch.Tensor:
        """Forward pass: pseudobulk [B, 31, 4796] → prediction [B].

        Constructs region_pseudobulk from pseudobulk OUTSIDE the PyroModule
        to maintain autograd graph connectivity, then calls forward_encoder_only
        with the pre-constructed region tensor.
        """
        from src.data.constants import PFC_REGION_IDX, N_REGIONS

        B = pseudobulk.shape[0]
        n_ct = pseudobulk.shape[1]
        n_genes = pseudobulk.shape[2]

        # Build region_pseudobulk outside PyroModule context
        region_pseudobulk = torch.zeros(
            B, N_REGIONS, n_ct, n_genes,
            device=pseudobulk.device, dtype=pseudobulk.dtype,
        )
        region_pseudobulk[:, PFC_REGION_IDX, :, :] = pseudobulk
        region_mask = torch.zeros(B, N_REGIONS, dtype=torch.bool, device=pseudobulk.device)
        region_mask[:, PFC_REGION_IDX] = True

        # Override region inputs in fixed kwargs
        kwargs = dict(self._fixed_kwargs)
        kwargs["region_pseudobulk"] = region_pseudobulk
        kwargs["region_mask"] = region_mask
        kwargs.pop("pseudobulk", None)  # don't pass pseudobulk — we passed region_pseudobulk

        output = self.model.forward_encoder_only(**kwargs)
        attended = output["attended"]  # [B, d_fused]

        # Bayesian head with guide median weights (manual F.linear for autograd)
        hw = self.head_weights
        h = F.gelu(F.linear(attended, hw["fc1.weight"], hw["fc1.bias"]))
        h = F.gelu(F.linear(h, hw["fc2.weight"], hw["fc2.bias"]))
        mean = F.linear(h, hw["fc_mean.weight"], hw["fc_mean.bias"])
        return mean.squeeze(-1)  # [B]


def _extract_head_weights(
    model: torch.nn.Module,
    guide: torch.nn.Module | None,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Extract prediction head weights for manual forward pass.

    For Bayesian models: uses guide's median for Bayesian layers,
    model's own parameters for non-Bayesian layers (fc_mean.bias, fc_log_std).
    """
    head = model.prediction_head
    weights = {}

    if guide is not None:
        import pyro.poutine
        median = pyro.poutine.block(guide.median)()

        # Map Pyro names to short names
        prefix = "cognitive_resilience_model.prediction_head."
        for pyro_name, value in median.items():
            short_name = pyro_name.replace(prefix, "")
            weights[short_name] = value.to(device)

        # Non-Bayesian params: get directly from model
        if "fc_mean.bias" not in weights:
            weights["fc_mean.bias"] = head.fc_mean.bias.data.to(device)
    else:
        # Deterministic head: use model params directly
        for name in ["fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias",
                      "fc_mean.weight", "fc_mean.bias"]:
            param = dict(head.named_parameters()).get(name)
            if param is not None:
                weights[name] = param.data.to(device)

    return weights


def compute_gene_attributions(
    model: torch.nn.Module,
    dataloader,
    device: str = "cuda:0",
    n_steps: int = 50,
    internal_batch_size: int = 4,
    show_progress: bool = True,
    guide: torch.nn.Module | None = None,
) -> dict[str, np.ndarray | list]:
    """Compute Integrated Gradients gene attributions for all subjects.

    Args:
        model: Trained CognitiveResilienceModel.
        dataloader: DataLoader yielding batches with pseudobulk + other fields.
        device: Torch device.
        n_steps: Number of interpolation steps for Integrated Gradients.
        internal_batch_size: Batch size for IG's internal forward passes.
        show_progress: Show tqdm progress bar.
        guide: Pyro guide for Bayesian models. Required for correct attributions.

    Returns:
        Dict with:
            - 'attributions': np.ndarray [n_subjects, n_cell_types, n_genes]
            - 'subject_ids': list[str]
            - 'predictions': np.ndarray [n_subjects]
    """
    dev = torch.device(device)
    model = model.to(dev)
    model.eval()

    if guide is not None:
        guide = guide.to(dev)

    head_weights = _extract_head_weights(model, guide, dev)
    wrapper = _PseudobulkForwardWrapper(model, head_weights)
    ig = IntegratedGradients(wrapper)

    all_attributions = []
    all_subject_ids = []
    all_predictions = []

    iterator = dataloader
    if show_progress:
        iterator = tqdm(dataloader, desc="Gene attribution", unit="batch")

    for batch in iterator:
        batch_dev = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch_dev[k] = v.to(dev)
            else:
                batch_dev[k] = v

        pseudobulk = batch_dev["pseudobulk"]  # [B, 31, 4796]
        wrapper.set_fixed_inputs(batch_dev)

        baseline = torch.zeros_like(pseudobulk)

        attr = ig.attribute(
            pseudobulk,
            baselines=baseline,
            n_steps=n_steps,
            internal_batch_size=internal_batch_size,
        )

        all_attributions.append(attr.detach().cpu().numpy())

        if "subject_ids" in batch:
            all_subject_ids.extend(batch["subject_ids"])

        with torch.no_grad():
            pred = wrapper(pseudobulk)
            all_predictions.append(pred.cpu().numpy())

    attributions = np.concatenate(all_attributions, axis=0)
    predictions = np.concatenate(all_predictions, axis=0)

    logger.info(
        "Computed gene attributions: %d subjects, shape %s",
        len(all_subject_ids), attributions.shape,
    )

    return {
        "attributions": attributions,
        "subject_ids": all_subject_ids,
        "predictions": predictions,
    }


def summarize_attributions(
    attributions: np.ndarray,
    gene_names: list[str] | np.ndarray,
    cell_type_names: list[str] | np.ndarray,
    top_k: int = 20,
) -> dict:
    """Summarize gene attributions into interpretable tables.

    Args:
        attributions: [n_subjects, n_cell_types, n_genes]
        gene_names: Gene name array [n_genes]
        cell_type_names: Cell type name array [n_cell_types]
        top_k: Number of top genes to return per cell type

    Returns:
        Dict with global_importance, per_cell_type, top_genes_per_cell_type, top_genes_global
    """
    abs_attr = np.abs(attributions)
    global_importance = abs_attr.mean(axis=(0, 1))
    per_ct_importance = abs_attr.mean(axis=0)

    top_per_ct = {}
    for ct_idx in range(attributions.shape[1]):
        ct_name = cell_type_names[ct_idx] if ct_idx < len(cell_type_names) else f"CT_{ct_idx}"
        ct_imp = per_ct_importance[ct_idx]
        top_idx = np.argsort(ct_imp)[::-1][:top_k]
        top_per_ct[ct_name] = [
            (gene_names[i] if i < len(gene_names) else f"gene_{i}", float(ct_imp[i]))
            for i in top_idx
        ]

    top_global_idx = np.argsort(global_importance)[::-1][:top_k]
    top_global = [
        (gene_names[i] if i < len(gene_names) else f"gene_{i}", float(global_importance[i]))
        for i in top_global_idx
    ]

    return {
        "global_importance": global_importance,
        "per_cell_type": per_ct_importance,
        "top_genes_per_cell_type": top_per_ct,
        "top_genes_global": top_global,
    }
