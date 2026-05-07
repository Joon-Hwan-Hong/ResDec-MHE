"""Per-subject counterfactual explanations (Wachter et al. 2017 Mode-A literal).

Minimizes ``L(x, λ) = ‖x − x_init‖² + λ · (f(x) − y_target)²`` with
adaptive λ doubling:

  * Start λ at ``--lambda-start`` (small → distance dominates).
  * Run inner gradient descent for up to ``--max-steps`` iterations
    with unit-norm L2 gradient clipping for stability at high λ.
  * If target not reached (``|f(x) − target| > --tol``), reset
    ``x ← x_init``, multiply λ by ``--lambda-mult`` (default 2.0),
    and retry.
  * Terminate at success OR when ``λ > --lambda-max``. The result
    records which λ was the converging / closest attempt.

Perturbations are restricted to the PFC slice of ``region_pseudobulk``
(31 cell types × 4,785 genes = 148 K dimensions). The other five
regions are held at their original (typically zero, since most
ROSMAP subjects are PFC-only) values, and the gradient is taken
only with respect to the PFC slice.

Two target modes are supported:
  - ``relative`` — ``y_target = y_init ± target_delta``: each subject
    is asked to move a fixed displacement in the opposite-regime
    direction, giving constant per-subject search difficulty (best
    for top-K aggregation across subjects).
  - ``absolute`` — ``y_target = ± target_delta``: each subject is
    asked to reach a fixed regime-side prediction value regardless
    of starting point, giving variable difficulty.

Performance: a single batched forward+backward through the model is
used for all selected subjects per Mode-A inner-loop step (Phase 1/2
optimization). Compared to the legacy per-subject loop:

  * Cell-branch is precomputed once per subject via
    ``CognitiveResilienceModel.compute_cell_emb_only`` and reused
    across all GD steps via ``forward_with_cached_cell_emb``
    (the cell branch is independent of ``region_pseudobulk``).
  * The PFC perturbation tensor is shared across the batch so each
    inner-loop step is one batched forward + one batched backward.
  * Per-subject ragged-stop frees a subject from further updates as
    soon as it hits ``|f(x) − target| ≤ tol`` (handled by
    ``find_counterfactual_mode_a_adaptive_batch``).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# P1.3b: enable dynamo's tensor-int specialization for cell_transformer's
# `int(counts.amax())`. This is required by Phase 1's eager-mode-bit-identical
# refactor in src/models/branches/cell_transformer.py:156. Setting this at
# import time guarantees the flag is in effect before any torch.compile call.
import torch._dynamo  # noqa: E402  (import ordering after torch)
torch._dynamo.config.capture_scalar_outputs = True

from omegaconf import OmegaConf  # noqa: E402

_WORKTREE_ROOT = Path(__file__).resolve().parents[3]
if str(_WORKTREE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKTREE_ROOT))

from src.analysis.counterfactual_resilience import (  # noqa: E402
    find_counterfactual_mode_a_adaptive,
    find_counterfactual_mode_a_adaptive_batch,
)
from src.data.constants import PFC_REGION_IDX  # noqa: E402
from src.data.datamodule import CognitiveResilienceDataModule  # noqa: E402
from src.data.splits import load_splits  # noqa: E402
from src.models.heads import DeterministicPredictionHead  # noqa: E402
from src.training.resdec_lightning_module import ResDecLightningModule  # noqa: E402
from src.utils.provenance import git_sha, pick_max_r2_ckpt  # noqa: E402

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Subject-batch merging (replicates collate_for_hgt_multiregion semantics over
# already-collated single-subject batches)
# ─────────────────────────────────────────────────────────────────────────────


def _stack_subject_batches(per_subject_batches: list[dict]) -> dict:
    """Merge a list of B single-subject batch dicts into one batched dict.

    Each input dict is the output of ``collate_for_hgt_multiregion`` with
    batch_size=1; the output mirrors the same collate, but with batch_size=B.
    Tensors are concatenated along the batch axis (or, for ragged cell/edge
    data, with proper offsetting) so the encoder accepts the merged dict
    without modification.

    Args:
        per_subject_batches: list of length B. Each entry is the
            ``(sid, batch_dict)``'s ``batch_dict`` from the val dataloader.

    Returns:
        dict with batched tensors:
          - region_pseudobulk: [B, n_regions, n_ct, n_gene]
          - region_mask: [B, n_regions]
          - pseudobulk: [B, n_ct, n_gene] (kept for completeness)
          - cell_type_mask: [B, n_ct]
          - cell_counts: [B, n_ct]
          - pathology: [B, n_pathology]
          - cognition: [B, 1]
          - metadata: [B, d_metadata] (if present in inputs)
          - ccc_edge_index: [2, E_total] with per-subject node offsets
          - ccc_edge_type: [E_total]
          - ccc_edge_attr: [E_total, edge_dim]
          - cell_data: [total_cells, n_genes]
          - cell_offsets: [B, n_ct + 1] (with per-subject offset shift)
          - subject_ids: list[str]
          - batch_size: int
    """
    if not per_subject_batches:
        raise ValueError("per_subject_batches must be non-empty")
    B = len(per_subject_batches)

    # Tensors that concatenate cleanly along dim=0 (each input is [1, ...]).
    stack_keys = (
        "region_pseudobulk", "region_mask", "pseudobulk",
        "cell_type_mask", "cell_counts", "pathology", "cognition",
    )
    out: dict = {}
    for key in stack_keys:
        if key in per_subject_batches[0] and per_subject_batches[0][key] is not None:
            out[key] = torch.cat(
                [b[key] for b in per_subject_batches], dim=0,
            )

    # Optional metadata (FiLM): concat along dim=0 if present.
    if "metadata" in per_subject_batches[0] and per_subject_batches[0]["metadata"] is not None:
        out["metadata"] = torch.cat(
            [b["metadata"] for b in per_subject_batches], dim=0,
        )

    # n_nodes_per_graph from the first batch (= n_cell_types). Each
    # subject's edge_index uses indices [0, n_nodes_per_graph); subject i's
    # indices must be shifted by i * n_nodes_per_graph so HGTEncoderTensor
    # treats them as separate graphs.
    s0_pseudobulk = per_subject_batches[0]["pseudobulk"]
    n_nodes_per_graph = int(s0_pseudobulk.shape[1])  # [1, n_ct, n_gene] -> n_ct

    edge_indices, edge_types, edge_attrs = [], [], []
    cell_data_chunks: list[torch.Tensor] = []
    cell_offsets_chunks: list[torch.Tensor] = []
    cumulative_cells = 0
    n_genes = int(s0_pseudobulk.shape[-1])

    for i, sb in enumerate(per_subject_batches):
        # Edges
        ei = sb.get("ccc_edge_index")
        if ei is not None and ei.numel() > 0:
            edge_indices.append(ei + i * n_nodes_per_graph)
            edge_types.append(sb["ccc_edge_type"])
            edge_attrs.append(sb["ccc_edge_attr"])

        # Cell data (variable-length flat tensor)
        cd = sb.get("cell_data")
        co = sb.get("cell_offsets")  # [1, n_ct + 1]
        # Per-subject offsets are [0, ..., total_cells_subj_i]; shift by
        # cumulative_cells so the merged offsets reference the merged cell_data.
        cell_offsets_chunks.append(co + cumulative_cells)
        if cd is not None and cd.shape[0] > 0:
            cell_data_chunks.append(cd)
        cumulative_cells += int(co[0, -1].item())

    if edge_indices:
        out["ccc_edge_index"] = torch.cat(edge_indices, dim=1)
        out["ccc_edge_type"] = torch.cat(edge_types, dim=0)
        out["ccc_edge_attr"] = torch.cat(edge_attrs, dim=0)
    else:
        device = s0_pseudobulk.device
        out["ccc_edge_index"] = torch.zeros(2, 0, dtype=torch.long, device=device)
        out["ccc_edge_type"] = torch.zeros(0, dtype=torch.long, device=device)
        out["ccc_edge_attr"] = torch.zeros(0, 1, device=device)

    if cell_data_chunks:
        out["cell_data"] = torch.cat(cell_data_chunks, dim=0)
    else:
        device = s0_pseudobulk.device
        out["cell_data"] = torch.empty(0, n_genes, device=device)

    out["cell_offsets"] = torch.cat(cell_offsets_chunks, dim=0)  # [B, n_ct+1]

    # Subject IDs as flat list (one per merged batch row).
    sids: list[str] = []
    for sb in per_subject_batches:
        sids.extend(list(sb.get("subject_ids", [])))
    out["subject_ids"] = sids
    out["batch_size"] = B
    return out


def _resolve_encoder(model: torch.nn.Module):
    """Return the underlying ResDecLightningModule, unwrapping torch.compile.

    ``torch.compile`` returns an ``OptimizedModule`` whose ``_orig_mod`` is the
    original lightning module. Attribute access (``.encoder``, ``.head``)
    flows through to ``_orig_mod`` automatically, but for explicit wiring
    (e.g. dispatching ``compute_cell_emb_only``) we want a clear handle.
    """
    return getattr(model, "_orig_mod", model)


# ─────────────────────────────────────────────────────────────────────────────
# Per-subject closure (legacy single-subject path, preserved for cross-
# validation tests). Refactored to use the cached cell-branch boundary so it's
# byte-comparable in semantics to the batched path.
# ─────────────────────────────────────────────────────────────────────────────


def _build_pfc_only_closures(
    model: torch.nn.Module,
    template_batch: dict,
    device: torch.device,
):
    """Build (f, grad_f) closures for PFC-slice-only perturbation (single subj).

    The perturbation vector ``x`` has shape (n_ct * n_gene,) and, when
    reshaped to (1, n_ct, n_gene), is injected into region PFC of a copy
    of the subject's ``region_pseudobulk`` tensor. Other regions retain
    whatever values they had in the template batch (usually zero for
    PFC-only subjects, real data for multi-region subjects). Gradient is
    taken only with respect to the PFC slice.

    Optimizations applied (P1.1, P1.6, P2.5):
      * Cell-branch precomputed via ``compute_cell_emb_only`` and reused
        through ``forward_with_cached_cell_emb`` — saves the entire
        CellTransformer ISAB stack per inner-loop step.
      * Pre-allocated ``rp_buf`` (constant non-PFC regions) cloned each
        step into a non-leaf tensor; the differentiable PFC slice is
        index-assigned into the clone so autograd flows back to xt_buf.
      * Pre-allocated no-grad scratch tensor for the inference path.
    """
    if "region_pseudobulk" not in template_batch:
        raise KeyError(
            "template_batch is missing 'region_pseudobulk'; this orchestrator "
            "requires the multi-region format (datamodule always provides this)"
        )
    orig_rp = template_batch["region_pseudobulk"]
    _, n_regions, n_ct, n_gene = orig_rp.shape
    pfc_shape = (1, n_ct, n_gene)
    n_features = int(np.prod(pfc_shape))

    inner = _resolve_encoder(model)

    # P1.1: cache cell-branch embedding ONCE under no_grad. The cell branch
    # depends only on cell_data/cell_offsets, which never change across GD
    # steps. The cached embedding is reused through forward_with_cached_cell_emb
    # for every step.
    with torch.no_grad():
        cell_emb_const = inner.encoder.compute_cell_emb_only(template_batch)

    # P1.6: pre-allocate the constant non-PFC region buffer once. Each step
    # clones this buffer (so it becomes a non-leaf tensor), then index-assigns
    # the differentiable PFC slice. The clone is essentially free vs the
    # alternative of cat()-ing each step (which also costs an extra
    # allocation but additionally fragments memory).
    rp_buf = orig_rp.clone()  # [1, n_regions, n_ct, n_gene]
    # Zero out the PFC slice in the buffer so the +xt assignment in the
    # closure is unambiguous (in case the original PFC slice was nonzero).
    rp_buf[:, PFC_REGION_IDX, :, :] = 0.0

    # Pre-allocate the perturbation tensor once with requires_grad=True.
    xt_buf = torch.zeros(pfc_shape, dtype=torch.float32, device=device,
                         requires_grad=True)
    # P2.5: pre-allocated no-grad scratch (avoids reallocating per inference call).
    xt_no_grad_buf = torch.zeros(pfc_shape, dtype=torch.float32, device=device)

    # Reusable batch dict — the closure swaps region_pseudobulk per call.
    template_batch_copy = dict(template_batch)

    # Pull metadata once. _get_metadata returns either batch["metadata"] or
    # a zero-fallback. We cache the resulting tensor so the head call uses a
    # consistent FiLM vector.
    metadata_const = inner._get_metadata(template_batch, batch_size=orig_rp.shape[0])

    def _run_head(rp: torch.Tensor) -> torch.Tensor:
        """Run encoder (with cached cell branch) + head; returns [B] prediction."""
        template_batch_copy["region_pseudobulk"] = rp
        enc_out = inner.encoder.forward_with_cached_cell_emb(
            template_batch_copy, cell_emb_const,
        )
        head_out = inner.head(enc_out["attended"], metadata_const)
        return head_out["prediction"].squeeze()

    def _forward_with_x(x_np: np.ndarray, requires_grad: bool):
        if requires_grad:
            if xt_buf.grad is not None:
                xt_buf.grad = None
            with torch.no_grad():
                xt_buf.data.copy_(
                    torch.from_numpy(x_np).to(device).reshape(pfc_shape)
                )
            # Build a non-leaf rp tensor by cloning the constant buffer and
            # index-assigning the differentiable PFC slice. The clone is the
            # graph root (non-leaf), so index assignment is allowed and
            # gradient flows back to xt_buf.
            rp = rp_buf.clone()
            rp[:, PFC_REGION_IDX, :, :] = xt_buf
            xt = xt_buf
        else:
            with torch.no_grad():
                xt_no_grad_buf.copy_(
                    torch.from_numpy(x_np).to(device).reshape(pfc_shape)
                )
            rp = rp_buf.clone()
            rp[:, PFC_REGION_IDX, :, :] = xt_no_grad_buf
            xt = xt_no_grad_buf
        y = _run_head(rp)
        return y, xt

    def f(x: np.ndarray) -> float:
        with torch.no_grad():
            y, _ = _forward_with_x(x, requires_grad=False)
            return float(y.detach().cpu().numpy())

    def grad_f(x: np.ndarray) -> np.ndarray:
        y, xt = _forward_with_x(x, requires_grad=True)
        y.backward()
        return xt.grad.reshape(-1).detach().cpu().numpy().astype(np.float64)

    def f_and_grad(x: np.ndarray) -> tuple[float, np.ndarray]:
        """Combined forward + backward; saves one forward pass per GD step."""
        y, xt = _forward_with_x(x, requires_grad=True)
        y.backward()
        return (
            float(y.detach().cpu().numpy()),
            xt.grad.reshape(-1).detach().cpu().numpy().astype(np.float64),
        )

    return f, grad_f, f_and_grad, n_features, pfc_shape


# ─────────────────────────────────────────────────────────────────────────────
# Batched closure (B subjects × n_features in one call). Drives the
# find_counterfactual_mode_a_adaptive_batch path.
# ─────────────────────────────────────────────────────────────────────────────


def _build_batched_pfc_only_closure(
    model: torch.nn.Module,
    merged_batch: dict,
    device: torch.device,
):
    """Build a single batched ``f_and_grad_batch`` for B subjects.

    The closure takes a numpy array ``X: [B, n_features]`` (each row is one
    subject's flattened PFC slice), runs ONE batched forward+backward, and
    returns ``(y: [B], g: [B, n_features])``. Per-subject gradients are
    isolated because each subject's prediction depends only on its own
    PFC slice (the shared cell branch is precomputed and detached from
    the autograd graph by virtue of being computed under no_grad).

    Returns:
        f_and_grad_batch, n_features, pfc_shape (per-subject [n_ct, n_gene])
    """
    if "region_pseudobulk" not in merged_batch:
        raise KeyError("merged_batch is missing 'region_pseudobulk'")
    orig_rp = merged_batch["region_pseudobulk"]  # [B, n_regions, n_ct, n_gene]
    B, n_regions, n_ct, n_gene = orig_rp.shape
    pfc_shape = (n_ct, n_gene)  # per-subject
    n_features = int(np.prod(pfc_shape))

    inner = _resolve_encoder(model)

    # Cell-branch embedding for ALL B subjects: [B, n_ct, d]. Computed once
    # under no_grad; reused for every CF step.
    with torch.no_grad():
        cell_emb_const = inner.encoder.compute_cell_emb_only(merged_batch)

    # Constant non-PFC region buffer for the batch.
    rp_buf = orig_rp.clone()  # [B, n_regions, n_ct, n_gene]
    rp_buf[:, PFC_REGION_IDX, :, :] = 0.0

    # Pre-allocated leaf tensor [B, n_ct, n_gene] for the differentiable PFC slice.
    xt_buf = torch.zeros(B, n_ct, n_gene, dtype=torch.float32, device=device,
                         requires_grad=True)

    # Reusable dict shell. We must keep the cell_data / cell_offsets / etc.
    # references intact so forward_with_cached_cell_emb can read them.
    template_batch_copy = dict(merged_batch)

    # Cache metadata once (FiLM input). Same vector reused every step.
    metadata_const = inner._get_metadata(merged_batch, batch_size=B)

    def f_and_grad_batch(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Batched forward + backward over B subjects.

        Args:
            X: ``[B, n_features]`` flattened PFC slices (one row per subject).

        Returns:
            (y: ``[B]``, g: ``[B, n_features]``) — predictions and per-subject
            gradients (rows are independent because each subject's pred only
            depends on its own X[i]).
        """
        # Reset gradient on the leaf buffer.
        if xt_buf.grad is not None:
            xt_buf.grad = None
        with torch.no_grad():
            xt_buf.data.copy_(
                torch.from_numpy(X.astype(np.float32))
                .to(device)
                .reshape(B, n_ct, n_gene)
            )
        # Build non-leaf rp by cloning the constant buffer and assigning the
        # differentiable PFC slice. Gradient flows back to xt_buf.
        rp = rp_buf.clone()
        rp[:, PFC_REGION_IDX, :, :] = xt_buf
        template_batch_copy["region_pseudobulk"] = rp

        enc_out = inner.encoder.forward_with_cached_cell_emb(
            template_batch_copy, cell_emb_const,
        )
        head_out = inner.head(enc_out["attended"], metadata_const)
        pred = head_out["prediction"].squeeze(-1) if head_out["prediction"].dim() == 2 else head_out["prediction"]
        # Sum-loss: each subject's pred depends only on its own X[i], so the
        # gradient of pred.sum() w.r.t. xt_buf is the per-row gradient
        # ∂pred[i]/∂X[i] (other rows are zero by construction).
        pred.sum().backward()

        y_np = pred.detach().cpu().numpy().astype(np.float64).reshape(-1)
        g_np = xt_buf.grad.detach().cpu().numpy().astype(np.float64).reshape(B, -1)
        return y_np, g_np

    return f_and_grad_batch, n_features, pfc_shape


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main():
    # P3.1: env-var defaults for all path-like args. Allows shell drivers to
    # set ``CF_OUT_DIR=...`` etc. without baking literals into wrappers (per
    # feedback_no_hardcoded_paths.md).
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "--config",
        default=os.environ.get("CF_CONFIG", "configs/resdec_mhe/canonical.yaml"),
    )
    p.add_argument("--fold", type=int, default=int(os.environ.get("CF_FOLD", 0)))
    p.add_argument(
        "--canonical-dir",
        default=os.environ.get(
            "CF_CANONICAL_DIR", "outputs/canonical/p5_canonical_seed42",
        ),
    )
    p.add_argument(
        "--splits-path",
        default=os.environ.get("CF_SPLITS_PATH", "outputs/splits.json"),
    )
    p.add_argument(
        "--residual-csv",
        default=os.environ.get(
            "CF_RESIDUAL_CSV",
            "outputs/canonical/interpretability/residual_per_subject.csv",
        ),
    )
    p.add_argument(
        "--device",
        default=os.environ.get("CF_DEVICE", "cuda:0"),
    )
    p.add_argument(
        "--out-dir",
        default=os.environ.get(
            "CF_OUT_DIR",
            "outputs/canonical/interpretability/counterfactuals",
        ),
    )
    p.add_argument(
        "--metadata-path",
        default=os.environ.get("CF_METADATA_PATH", None),
        help=(
            "Override cfg.data.metadata_path (parent dir of metadata.csv). "
            "Required when configs/default.yaml has the sanitized placeholder."
        ),
    )
    p.add_argument(
        "--precomputed-dir",
        default=os.environ.get("CF_PRECOMPUTED_DIR", None),
        help=(
            "Override cfg.data.precomputed_dir. Required when configs/default.yaml "
            "has the sanitized placeholder."
        ),
    )
    p.add_argument("--n-resilient", type=int, default=10)
    p.add_argument("--n-vulnerable", type=int, default=10)
    p.add_argument("--max-steps", type=int, default=1000,
                   help="Inner GD steps per λ attempt.")
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--tol", type=float, default=1e-3,
                   help="Convergence tolerance |f(x) - target| <= tol.")
    p.add_argument("--lambda-start", type=float, default=1e-3,
                   help="Initial λ (small = distance-dominant).")
    p.add_argument("--lambda-max", type=float, default=1e3,
                   help="Terminate search when λ exceeds this value.")
    p.add_argument("--lambda-mult", type=float, default=2.0,
                   help="λ multiplier per adaptive step (Wachter: 2.0).")
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--target-delta", type=float, default=0.5)
    p.add_argument(
        "--target-mode", choices=["relative", "absolute"], default="relative",
        help=(
            "relative: y_target = y_init ± target_delta (regime-direction; "
            "constant per-subject search difficulty, better for top-K "
            "aggregation). "
            "absolute: y_target = ± target_delta (regime-direction; "
            "variable per-subject difficulty, answers 'how hard to push "
            "THIS subject across the y=0 decision line?')."
        ),
    )
    p.add_argument("--seed", type=int, default=42,
                   help="Manual seed for torch / numpy (recorded in JSON).")
    p.add_argument("--no-compile", action="store_true",
                   help="Disable torch.compile (use plain eager mode).")
    p.add_argument(
        "--per-subject", action="store_true",
        help=(
            "Use legacy per-subject loop instead of batched API. Used by "
            "cross-validation tests; the batched path is canonical."
        ),
    )
    p.add_argument(
        "--record-trajectory", action="store_true",
        help=(
            "If set, record per-subject (lambda, residual_at_end_of_inner_loop) "
            "tuples per λ doubling and serialize them in the per-subject "
            "result dict. Default off to keep the JSON compact."
        ),
    )
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    # Apply seed (P2.4 — actually use --seed instead of leaving it unused).
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = OmegaConf.merge(
        OmegaConf.load("configs/default.yaml"),
        OmegaConf.load(args.config),
    )
    OmegaConf.set_struct(cfg, False)
    cfg.model.head.type = "deterministic"
    cfg.data.fold = int(args.fold)
    if args.metadata_path is not None:
        cfg.data.metadata_path = args.metadata_path
    if args.precomputed_dir is not None:
        cfg.data.precomputed_dir = args.precomputed_dir

    fold_dir = Path(args.canonical_dir) / f"fold{args.fold}"
    ckpt_path = pick_max_r2_ckpt(fold_dir / "checkpoints")
    logger.info("fold %d: loading %s", args.fold, ckpt_path.name)

    splits = load_splits(str(args.splits_path))
    metadata = pd.read_csv(Path(cfg.data.metadata_path) / "metadata.csv")
    dm = CognitiveResilienceDataModule(
        config=cfg, metadata=metadata, splits=splits,
        fold_idx=args.fold,
        precomputed_dir=cfg.data.precomputed_dir,
        adata=None,
    )
    dm.setup(stage="fit")
    dm.batch_size = 1  # per-subject perturbation; avoids flattened-edge-tensor slicing issues.

    model = ResDecLightningModule.load_from_checkpoint(
        str(ckpt_path), config=cfg, map_location="cpu",
    ).to(device).eval()

    # P1.3b: assert deterministic head — Bayesian heads use pyro.sample which
    # is incompatible with the gradient flow this orchestrator relies on.
    assert isinstance(model.encoder.prediction_head, DeterministicPredictionHead), (
        f"CF orchestrator requires a deterministic prediction head; "
        f"got {type(model.encoder.prediction_head).__name__}"
    )

    # Optimization B: torch.compile the model in eval mode. Saves ~1.1-1.3×
    # on forward+backward via op fusion. Numerically identical to ~1e-6.
    # mode="default" instead of "reduce-overhead" because cudagraphs (used
    # by reduce-overhead) is incompatible with FiLM's gamma_proj(metadata)
    # tensor-reuse pattern.
    # Skip via --no-compile when debugging.
    if not args.no_compile:
        try:
            model = torch.compile(model, mode="default", fullgraph=False)
            logger.info("torch.compile enabled (mode=default)")
        except Exception as exc:
            logger.warning("torch.compile failed (%s); using uncompiled model", exc)

    # Residuals for subject selection.
    res_df = pd.read_csv(args.residual_csv)
    id_col = (
        "ROSMAP_IndividualID" if "ROSMAP_IndividualID" in res_df.columns
        else res_df.columns[0]
    )
    res_df = res_df.rename(columns={id_col: "subject_id"})
    res_df["subject_id"] = res_df["subject_id"].astype(str)
    val_subject_ids: set[str] = set()
    for batch in dm.val_dataloader():
        val_subject_ids.update(str(s) for s in batch["subject_ids"])
    val_res = res_df[res_df["subject_id"].isin(val_subject_ids)].copy()
    val_res = val_res[np.isfinite(val_res["residual"])]
    top_res = val_res.nlargest(args.n_resilient, "residual")["subject_id"].tolist()
    top_vul = val_res.nsmallest(args.n_vulnerable, "residual")["subject_id"].tolist()
    regime_map: dict[str, str] = {**{s: "resilient" for s in top_res},
                                   **{s: "vulnerable" for s in top_vul}}
    logger.info(
        "selected %d resilient + %d vulnerable val-fold subjects for CF search",
        len(top_res), len(top_vul),
    )

    subject_batches: list[tuple[str, dict]] = []
    for batch in dm.val_dataloader():
        sids = list(batch["subject_ids"])
        assert len(sids) == 1, f"expected batch_size=1; got {len(sids)}"
        sid_str = str(sids[0])
        if sid_str not in regime_map:
            continue
        per = {
            k: (v.to(device) if torch.is_tensor(v) else v)
            for k, v in batch.items()
        }
        subject_batches.append((sid_str, per))

    # Preserve a stable order keyed by selection order in `regime_map` so the
    # JSON has a deterministic per-subject layout independent of dataloader
    # iteration order on this fold.
    target_order = [s for s in (top_res + top_vul) if s in dict(subject_batches)]
    by_sid = dict(subject_batches)
    subject_batches = [(s, by_sid[s]) for s in target_order]

    if args.per_subject:
        results = _run_per_subject(
            model, subject_batches, regime_map, device, args,
        )
    else:
        results = _run_batched(
            model, subject_batches, regime_map, device, args,
        )

    n_features, pfc_shape = _shape_summary(subject_batches[0][1]) if subject_batches else (None, None)

    # _run_batched / _run_per_subject both return {"results": [...], "__elapsed_min__": ...}.
    results_list = results["results"]
    elapsed_min = results["__elapsed_min__"]

    summary = {
        "method": "Wachter et al. 2017 Mode-A literal, adaptive λ doubling",
        "loss": "L(x, λ) = ‖x - x_init‖² + λ · (f(x) - y_target)²",
        "target_mode": args.target_mode,
        "fold": args.fold,
        "ckpt": str(ckpt_path),
        "n_features_per_subject": n_features,
        "pfc_shape": list(pfc_shape) if pfc_shape else None,
        "n_subjects_processed": len(results_list),
        "n_resilient_targeted": len(top_res),
        "n_vulnerable_targeted": len(top_vul),
        "max_steps": args.max_steps,
        "lr": args.lr,
        "tol": args.tol,
        "lambda_start": args.lambda_start,
        "lambda_max": args.lambda_max,
        "lambda_mult": args.lambda_mult,
        "target_delta": args.target_delta,
        "top_k_per_subject": args.top_k,
        "seed": args.seed,
        "elapsed_min": elapsed_min,
        "git_commit": git_sha(_WORKTREE_ROOT),
        "results": results_list,
    }
    out_path = out_dir / f"counterfactuals_fold{args.fold}.json"
    out_path.write_text(json.dumps(summary, indent=2))
    logger.info(
        "wrote %s (%.1f min total)", out_path, summary["elapsed_min"],
    )
    return 0


def _shape_summary(template_batch: dict) -> tuple[int, tuple[int, int, int]]:
    """Compute (n_features, pfc_shape) for the JSON summary."""
    rp = template_batch["region_pseudobulk"]
    _, _, n_ct, n_gene = rp.shape
    pfc_shape = (1, n_ct, n_gene)
    return int(n_ct * n_gene), pfc_shape


def _build_target_y(y_init: float, regime: str, args) -> float:
    """Translate (regime, args.target_mode) → numeric target_y."""
    if args.target_mode == "absolute":
        return -args.target_delta if regime == "resilient" else args.target_delta
    return y_init - args.target_delta if regime == "resilient" else y_init + args.target_delta


def _make_top_k(cf, top_k: int) -> list[dict]:
    delta = np.abs(cf.x_cf - cf.x_init)
    top_k_idx = np.argsort(delta)[::-1][:top_k]
    return [
        {
            "feature_idx": int(i),
            "abs_delta": float(delta[i]),
            "x_init": float(cf.x_init[i]),
            "x_cf": float(cf.x_cf[i]),
        }
        for i in top_k_idx
    ]


def _run_per_subject(
    model, subject_batches, regime_map, device, args,
) -> dict:
    """Legacy single-subject loop. Used for cross-validation only."""
    results: list[dict] = []
    t0 = time.time()
    for idx, (sid, template_batch) in enumerate(subject_batches):
        if "region_pseudobulk" not in template_batch or template_batch["region_pseudobulk"] is None:
            logger.warning("subject %s: no region_pseudobulk; skipping", sid)
            continue
        f, grad_f, f_and_grad, _n_features, _pfc_shape = _build_pfc_only_closures(
            model, template_batch, device,
        )
        x_init = (
            template_batch["region_pseudobulk"][:, PFC_REGION_IDX, :, :]
            .detach().cpu().numpy().reshape(-1).astype(np.float64)
        )
        y_init = f(x_init)
        regime = regime_map[sid]
        target_y = _build_target_y(y_init, regime, args)
        t_s = time.time()
        cf = find_counterfactual_mode_a_adaptive(
            f, grad_f, x_init, target_y,
            lr=args.lr, max_steps=args.max_steps, tol=args.tol,
            lambda_start=args.lambda_start, lambda_max=args.lambda_max,
            lambda_mult=args.lambda_mult,
            f_and_grad=f_and_grad,
            record_trajectory=args.record_trajectory,
        )
        # P2.2: emit ``gap`` and ``lambda_max_attempted`` alongside the legacy
        # ``lambda_used`` for downstream compat.
        result_dict = {
            "subject_id": sid,
            "regime": regime,
            "y_init": cf.y_init,
            "y_cf": cf.y_cf,
            "target_y": cf.target_y,
            "success": cf.success,
            "n_steps_used": cf.n_steps_used,
            "l2_distance": cf.l2_distance,
            "lambda_used": cf.lambda_best,
            "lambda_best": cf.lambda_best,
            "lambda_max_attempted": cf.lambda_max_attempted,
            "gap": cf.gap,
            "elapsed_s": round(time.time() - t_s, 1),
            "top_k_features": _make_top_k(cf, args.top_k),
        }
        if args.record_trajectory:
            # list of (lam, residual_at_end_of_inner_loop) per λ doubling
            result_dict["trajectory"] = [
                [float(lam), float(res)] for (lam, res) in cf.trajectory
            ]
        results.append(result_dict)
        logger.info(
            "[%d/%d] %s (%s): y=%+.3f -> %+.3f (target %+.3f), "
            "steps=%d, L2=%.3g, lam=%.3g, gap=%.3g, success=%s, t=%.1fs",
            idx + 1, len(subject_batches), sid, regime,
            cf.y_init, cf.y_cf, cf.target_y, cf.n_steps_used,
            cf.l2_distance, cf.lambda_best, cf.gap, cf.success,
            results[-1]["elapsed_s"],
        )
    return {"results": results, "__elapsed_min__": round((time.time() - t0) / 60, 2)}


def _run_batched(
    model, subject_batches, regime_map, device, args,
) -> dict:
    """Canonical batched path: ONE forward+backward per CF inner-loop step.

    P1.2 — drives ``find_counterfactual_mode_a_adaptive_batch`` with a single
    batched closure that processes all B subjects in a single forward+backward.
    """
    if not subject_batches:
        return {"results": [], "__elapsed_min__": 0.0}

    # Merge per-subject batches into one batched dict.
    per_subject_dicts = [b for _, b in subject_batches]
    sids_ordered = [sid for sid, _ in subject_batches]
    merged = _stack_subject_batches(per_subject_dicts)
    # Ensure all tensors live on `device`. _stack_subject_batches preserves the
    # input tensors' devices; per-subject batches were already moved to device
    # at collection time, so merged inherits that placement.

    f_and_grad_batch, _n_features, pfc_shape = _build_batched_pfc_only_closure(
        model, merged, device,
    )

    # Initial x: each subject's PFC slice flattened. Stack into [B, n_features].
    rp = merged["region_pseudobulk"]
    B = rp.shape[0]
    x_init_batch = (
        rp[:, PFC_REGION_IDX, :, :]
        .detach().cpu().numpy().reshape(B, -1).astype(np.float64)
    )

    # Initial y: ONE batched forward to determine relative-mode targets.
    # We call f_and_grad_batch directly (it manages its own requires_grad
    # via the closure-internal leaf buffer); the gradient output is unused
    # at this stage but the call is cheap. This y_init is then re-derived
    # inside find_counterfactual_mode_a_adaptive_batch's first invocation,
    # which is fine — both calls are at the same x_init so the model is
    # bit-identical and only one inner-loop step's worth of work is wasted.
    y_init_batch, _g = f_and_grad_batch(x_init_batch)

    target_y_batch = np.zeros(B, dtype=np.float64)
    for i, sid in enumerate(sids_ordered):
        target_y_batch[i] = _build_target_y(
            float(y_init_batch[i]), regime_map[sid], args,
        )

    t0 = time.time()
    logger.info(
        "starting batched CF search: B=%d, n_features=%d, max_steps=%d, "
        "lambda in [%.3g, %.3g]",
        B, _n_features, args.max_steps, args.lambda_start, args.lambda_max,
    )
    cf_results = find_counterfactual_mode_a_adaptive_batch(
        f_and_grad_batch, x_init_batch, target_y_batch,
        lr=args.lr, max_steps=args.max_steps, tol=args.tol,
        lambda_start=args.lambda_start, lambda_max=args.lambda_max,
        lambda_mult=args.lambda_mult,
        record_trajectory=args.record_trajectory,
    )
    elapsed_min = round((time.time() - t0) / 60, 2)
    logger.info("batched CF search complete: %.1f min", elapsed_min)

    results: list[dict] = []
    for i, cf in enumerate(cf_results):
        sid = sids_ordered[i]
        regime = regime_map[sid]
        result_dict = {
            "subject_id": sid,
            "regime": regime,
            "y_init": cf.y_init,
            "y_cf": cf.y_cf,
            "target_y": cf.target_y,
            "success": cf.success,
            "n_steps_used": cf.n_steps_used,
            "l2_distance": cf.l2_distance,
            "lambda_used": cf.lambda_best,
            "lambda_best": cf.lambda_best,
            "lambda_max_attempted": cf.lambda_max_attempted,
            "gap": cf.gap,
            "elapsed_s": None,  # batched: per-subject wall time not separable
            "top_k_features": _make_top_k(cf, args.top_k),
        }
        if args.record_trajectory:
            # list of (lam, residual_at_end_of_inner_loop) per λ doubling
            result_dict["trajectory"] = [
                [float(lam), float(res)] for (lam, res) in cf.trajectory
            ]
        results.append(result_dict)
        logger.info(
            "[%d/%d] %s (%s): y=%+.3f -> %+.3f (target %+.3f), "
            "steps=%d, L2=%.3g, lam=%.3g, gap=%.3g, success=%s",
            i + 1, len(cf_results), sid, regime,
            cf.y_init, cf.y_cf, cf.target_y, cf.n_steps_used,
            cf.l2_distance, cf.lambda_best, cf.gap, cf.success,
        )
    return {"results": results, "__elapsed_min__": elapsed_min}


if __name__ == "__main__":
    raise SystemExit(main())
