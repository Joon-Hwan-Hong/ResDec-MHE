"""GPU integration of AttnLRP / GMAR / GAF for ResDec-MHE pathology-stratified attention.

For each canonical fold checkpoint, runs forward + backward to obtain per-subject
attention weights and gradients on the PathologyStratifiedAttention layer, then
applies AttnLRP / GMAR / GAF reference implementations from
``src/analysis/attention_attribution.py`` to produce per-CT relevance scores.

Why a monkey-patch is required
------------------------------
The canonical ``PathologyStratifiedAttention.forward`` (line 191 in
``src/models/fusion/pathology_attention.py``) wraps the attention re-compute in
``torch.no_grad()`` to avoid redundant compute during inference. For AttnLRP /
GMAR / GAF we need gradients to flow through softmax + matmul. We therefore
monkey-patch ``forward`` on the loaded *instance* (not the class) so the
canonical source remains unchanged. The canonical pathology_attention path is
read-only at attribution time.

Output (default ``outputs/canonical/interpretability/attention_attribution/``):
  - per_subject_attribution.npz       — keys: subject_ids [N], attnlrp [N, C],
                                          gmar [N, C], gaf_af [N, C], gaf_gf [N, C],
                                          gaf_agf [N, C], fold [N]
  - attention_attribution_summary.json
                                      — per-CT mean importance per method;
                                          rank-1 CT per method; cross-method top-5

Usage
-----
    PYTHONPATH=<worktree-root> \\
    CUDA_VISIBLE_DEVICES=0 \\
    uv run python scripts/resdec_mhe/interpretability/run_attention_attribution.py \\
        --pred-root outputs/canonical/p5_canonical_seed42 \\
        --out-dir outputs/canonical/interpretability/attention_attribution
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from tqdm import tqdm

_WORKTREE_ROOT = Path(__file__).resolve().parents[3]
if not (_WORKTREE_ROOT / "src").is_dir():
    raise RuntimeError(
        f"sys.path bootstrap failed: {_WORKTREE_ROOT}/src not found; "
        "set PYTHONPATH=<worktree-root>."
    )
sys.path.insert(0, str(_WORKTREE_ROOT))

from src.analysis.attention_attribution import attnlrp_softmax
from src.utils.cell_types import pad_cell_type_names
from src.utils.provenance import git_sha, pick_max_r2_ckpt
from src.data.constants import CELL_TYPE_ORDER, N_REGIONS, PFC_REGION_IDX
from src.data.datamodule import CognitiveResilienceDataModule
from src.data.splits import load_splits
from src.training.resdec_lightning_module import ResDecLightningModule

logger = logging.getLogger(__name__)


def _patch_pathology_attention_for_grad(pa_module: torch.nn.Module) -> None:
    """Monkey-patch ``pa_module.forward`` to expose attention scores in the autograd graph.

    The canonical implementation of ``PathologyStratifiedAttention.forward``
    (a) computes ``attended`` via SDPA, and (b) re-derives ``attention_weights``
    in a separate ``torch.no_grad()`` block. For attribution we must (i) keep
    softmax + matmul in the graph, and (ii) stash intermediates (scores, V) on
    the instance for AttnLRP. We replace the instance's ``forward`` method with
    one that uses an explicit einsum + softmax path (numerically equivalent to
    SDPA up to FlashAttention reorderings, sufficient for IG-class attribution).
    """
    def grad_forward(self, cell_type_embeddings, path_emb,
                     cell_type_mask=None, return_attention_weights=True):
        B, C, D = cell_type_embeddings.shape
        # Q / K / V (same as canonical forward)
        query = self.query_generator(path_emb).view(
            B, self.n_heads, 1, self.d_head)
        keys = self.key_proj(cell_type_embeddings).view(
            B, C, self.n_heads, self.d_head).permute(0, 2, 1, 3)
        values = self.value_proj(cell_type_embeddings).view(
            B, C, self.n_heads, self.d_head).permute(0, 2, 1, 3)
        # Pathology-additive bias
        path_emb_expanded = path_emb.unsqueeze(1).expand(-1, self.n_cell_types, -1)
        bias_input = torch.cat([path_emb_expanded, cell_type_embeddings], dim=-1)
        bias = self.pathology_bias(bias_input)
        attn_bias = bias.permute(0, 2, 1).unsqueeze(2)
        all_masked = torch.zeros(
            B, dtype=torch.bool, device=cell_type_embeddings.device)
        if cell_type_mask is not None:
            mask = cell_type_mask.unsqueeze(1).unsqueeze(2)
            attn_bias = attn_bias.masked_fill(~mask, float('-inf'))
            all_masked = ~cell_type_mask.any(dim=1)
            if all_masked.any():
                all_masked_expanded = all_masked.view(-1, 1, 1, 1).expand_as(
                    attn_bias)
                attn_bias = attn_bias.masked_fill(all_masked_expanded, -1e9)
        # Scores + softmax IN-GRAPH (replaces SDPA + no_grad re-compute).
        # Numerical equivalence to SDPA path: softmax in float32 + masked_fill
        # matches canonical pathology_attention.py:194-198 to within ~1e-6
        # absolute tolerance. Drift expected from FlashAttention reorderings
        # is negligible for AttnLRP relevance rankings.
        # The masked_fill on attended/attention_weights MUST be applied to a
        # tensor that is on the path from softmax → attended; otherwise the
        # downstream stash is detached from the autograd graph (manifested as
        # "One of the differentiated Tensors appears to not have been used in
        # the graph"). We therefore apply masked_fill on the [B, H, C] slice
        # BEFORE building attended, so attended and the stash share one node.
        scores = torch.einsum('bhqd,bhkd->bhqk', query, keys) / (self.d_head ** 0.5)
        scores = scores + attn_bias  # [B, H, 1, C]
        scores_2d = scores.squeeze(2)  # [B, H, C]
        attention_weights = F.softmax(
            scores_2d.float(), dim=-1).to(values.dtype)  # [B, H, C]
        if cell_type_mask is not None and all_masked.any():
            mask_3d = all_masked.unsqueeze(-1).unsqueeze(-1).expand_as(attention_weights)
            attention_weights = attention_weights.masked_fill(mask_3d, 0.0)
        # Attended via in-graph attention × V (using the [B, H, C] weights).
        attended = torch.einsum(
            'bhk,bhkd->bhd', attention_weights, values
        ).reshape(B, self.d_fused)
        attended = self.out_proj(attended)
        if cell_type_mask is not None and all_masked.any():
            attended = attended.masked_fill(
                all_masked.unsqueeze(-1).expand_as(attended), 0.0)
        # Stash for post-hoc attribution. attention_weights is in-graph; scores
        # is also in-graph (used to compute attention_weights via softmax).
        self._last_scores = scores_2d  # [B, H, C]
        self._last_attention_weights = attention_weights  # [B, H, C]
        self._last_values = values  # [B, H, C, d_head]
        return attended, attention_weights

    pa_module.forward = types.MethodType(grad_forward, pa_module)


def _per_subject_attribution(
    A: np.ndarray,        # [H, C]   softmax outputs
    grad_A: np.ndarray,   # [H, C]   d output / d A
    scores: np.ndarray,   # [H, C]   pre-softmax scores
) -> dict[str, np.ndarray]:
    """Return per-CT vectors for AttnLRP (softmax rule), GMAR (L2 head weights),
    and GAF (AF / GF / AGF information-tensor variants).

    Outputs are 1D arrays of length C (one entry per cell type).
    """
    H = A.shape[0]
    # Clamp -inf scores at masked CTs (attn_bias = -inf for absent cell types
    # or -1e9 for fully-masked subjects). These positions have A=0 and grad=0
    # by construction; -inf * 0 = NaN in IEEE float, so we replace -inf with 0
    # before AttnLRP. The relevance at those positions stays 0.
    scores_safe = np.where(np.isfinite(scores), scores, 0.0)
    # AttnLRP: R^l = A * grad_A (canonical relevance of softmax output);
    # propagate through softmax to scores: R^{l-1}_i = x_i (R^l_i - s_i Σ_j R^l_j)
    R_l = A * grad_A  # [H, C]
    R_input = attnlrp_softmax(R_l, A, scores_safe)  # [H, C]
    # Force masked positions (where A is at-or-near zero across all heads) to
    # 0 relevance. Threshold 1e-10 is robust to float32 underflow of
    # softmax(-inf) ≈ subnormal vs exact 0; both should be treated as masked.
    R_input = np.where(A < 1e-10, 0.0, R_input)
    attnlrp_per_ct = R_input.mean(axis=0)  # mean over heads → [C]

    # GMAR Algorithm 1 (L2 norm) — DEVIATION from literal Algorithm 1 noted:
    # The literal GMAR algorithm operates on the full multi-token attention
    # matrix [H, N, N] and applies a weighted rollout across L layers. Here
    # we have a single-query attention [H, C] (pathology query attending over
    # C cell types) and only ONE attention layer (PathologyStratifiedAttention),
    # so the rollout step degenerates to identity and we apply only the
    # gradient-based per-head L2 weighting + weighted-mean across heads.
    # Output is therefore "GMAR-style head-weighted attention" (single-query,
    # single-layer adaptation), NOT literal Algorithm 1. Same per-head L2
    # weighting as the paper; only the rollout step is simplified-by-degeneracy.
    per_head_l2 = np.sqrt(np.sum(grad_A ** 2, axis=-1))  # [H]
    denom = per_head_l2.sum()
    w_h = per_head_l2 / denom if denom > 1e-12 else np.full(H, 1.0 / H)
    gmar_per_ct = (w_h[:, None] * A).sum(axis=0)

    # GAF AF / GF / AGF — compute per-CT importance directly
    # (full information-tensor not applicable since query length = 1)
    af_per_ct = A.mean(axis=0)
    gf_per_ct = np.maximum(grad_A, 0).mean(axis=0)
    agf_per_ct = np.maximum(A * grad_A, 0).mean(axis=0)

    return {
        "attnlrp": attnlrp_per_ct.astype(np.float32),
        "gmar": gmar_per_ct.astype(np.float32),
        "gaf_af": af_per_ct.astype(np.float32),
        "gaf_gf": gf_per_ct.astype(np.float32),
        "gaf_agf": agf_per_ct.astype(np.float32),
    }


def attribute_one_fold(args: argparse.Namespace, fold: int,
                       device: torch.device) -> dict:
    cfg = OmegaConf.merge(
        OmegaConf.load("configs/default.yaml"),
        OmegaConf.load(args.config),
    )
    OmegaConf.set_struct(cfg, False)
    cfg.model.head.type = "deterministic"
    cfg.data.fold = int(fold)
    if getattr(args, "metadata_path", None) is not None:
        cfg.data.metadata_path = str(args.metadata_path)
    if getattr(args, "precomputed_dir", None) is not None:
        cfg.data.precomputed_dir = str(args.precomputed_dir)

    fold_dir = Path(args.pred_root) / f"fold{fold}"
    ckpt_path = pick_max_r2_ckpt(fold_dir / "checkpoints")
    logger.info("fold %d: loading %s", fold, ckpt_path.name)

    splits = load_splits(str(args.splits_path))
    metadata_csv = pd.read_csv(Path(cfg.data.metadata_path) / "metadata.csv")
    dm = CognitiveResilienceDataModule(
        config=cfg, metadata=metadata_csv, splits=splits,
        fold_idx=fold,
        precomputed_dir=cfg.data.precomputed_dir,
        adata=None,
    )
    dm.setup(stage="fit")

    model = ResDecLightningModule.load_from_checkpoint(
        str(ckpt_path), config=cfg, map_location="cpu",
    ).to(device).eval()
    model = model.float()

    _patch_pathology_attention_for_grad(model.encoder.pathology_attention)

    sids_all: list[str] = []
    out: dict[str, list[np.ndarray]] = {
        "attnlrp": [], "gmar": [], "gaf_af": [], "gaf_gf": [], "gaf_agf": [],
    }

    val_loader = dm.val_dataloader()
    for batch in tqdm(val_loader, desc=f"fold {fold} attn-attr", unit="batch"):
        sids = list(batch["subject_ids"])
        sids_all.extend(sids)
        batch_d = {
            k: (v.to(device).float()
                if (torch.is_tensor(v) and v.is_floating_point())
                else v.to(device) if torch.is_tensor(v) else v)
            for k, v in batch.items()
        }
        pseudobulk = batch_d["pseudobulk"]
        B, n_ct, n_genes = pseudobulk.shape
        region_pseudobulk = torch.zeros(
            B, N_REGIONS, n_ct, n_genes, device=device, dtype=pseudobulk.dtype)
        region_pseudobulk[:, PFC_REGION_IDX, :, :] = pseudobulk
        region_mask = torch.zeros(B, N_REGIONS, dtype=torch.bool, device=device)
        region_mask[:, PFC_REGION_IDX] = True

        kwargs = {
            "ccc_edge_index": batch_d.get("ccc_edge_index"),
            "ccc_edge_type": batch_d.get("ccc_edge_type"),
            "ccc_edge_attr": batch_d.get("ccc_edge_attr"),
            "cell_type_mask": batch_d.get("cell_type_mask"),
            "pathology": batch_d.get("pathology"),
            "cell_data": batch_d.get("cell_data"),
            "cell_offsets": batch_d.get("cell_offsets"),
            "region_pseudobulk": region_pseudobulk,
            "region_mask": region_mask,
        }
        with torch.enable_grad():
            enc_out = model.encoder.forward_encoder_only(**kwargs)
            z = enc_out["attended"]
            metadata = torch.zeros(
                B, model._d_metadata, device=device, dtype=z.dtype)
            head_out = model.head(z, metadata)
            scalar = head_out["prediction"].sum()

        pa = model.encoder.pathology_attention
        A_full = pa._last_attention_weights  # [B, H, C]
        scores = pa._last_scores  # [B, H, C]
        grad_A = torch.autograd.grad(scalar, A_full, retain_graph=False)[0]
        A_np = A_full.detach().cpu().numpy()
        grad_np = grad_A.detach().cpu().numpy()
        scores_np = scores.detach().cpu().numpy()  # [B, H, C]

        for b in range(B):
            res = _per_subject_attribution(A_np[b], grad_np[b], scores_np[b])
            for k in out:
                out[k].append(res[k])

    return {
        "subject_ids": np.array(sids_all, dtype=object),
        **{k: np.stack(v).astype(np.float32) for k, v in out.items()},
        "fold": np.full(len(sids_all), fold, dtype=np.int32),
    }


def summarize(per_subject: dict[str, np.ndarray],
              ct_names: list[str], top_k: int = 10) -> dict:
    """Aggregate per-subject [N, C] attribution matrices into per-CT ranks per method."""
    methods = ["attnlrp", "gmar", "gaf_af", "gaf_gf", "gaf_agf"]
    summary: dict[str, dict] = {}
    for m in methods:
        arr = np.abs(per_subject[m]) if m == "attnlrp" else per_subject[m]
        per_ct_mean = arr.mean(axis=0)  # [C]
        order = np.argsort(-per_ct_mean)
        summary[m] = {
            "rank_by_mean_importance": [
                {"cell_type": ct_names[c], "mean_importance": float(per_ct_mean[c])}
                for c in order
            ],
            "top_1_cell_type": ct_names[int(order[0])],
            "top_k_cell_types": [ct_names[int(c)] for c in order[:top_k]],
        }
    return summary


def main(args: argparse.Namespace) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    sids_per: list[np.ndarray] = []
    out_per_method: dict[str, list[np.ndarray]] = {
        "attnlrp": [], "gmar": [], "gaf_af": [], "gaf_gf": [], "gaf_agf": [],
    }
    fold_per: list[np.ndarray] = []
    for f in range(int(args.n_folds)):
        d = attribute_one_fold(args, f, device)
        sids_per.append(d["subject_ids"])
        fold_per.append(d["fold"])
        for k in out_per_method:
            out_per_method[k].append(d[k])
        logger.info("fold %d: shape=%s", f, d["attnlrp"].shape)

    sids = np.concatenate(sids_per)
    folds = np.concatenate(fold_per)
    merged = {k: np.concatenate(v, axis=0) for k, v in out_per_method.items()}

    n_subj, n_ct = merged["attnlrp"].shape
    logger.info("Total: %d subjects × %d cell types per method", n_subj, n_ct)

    out_npz = out_dir / "per_subject_attribution.npz"
    np.savez(out_npz, subject_ids=sids, fold=folds, **merged)
    logger.info("Wrote %s", out_npz)

    ct_names = pad_cell_type_names(CELL_TYPE_ORDER, n_ct)

    summary = summarize(merged, ct_names, top_k=int(args.top_k))
    summary_path = out_dir / "attention_attribution_summary.json"
    summary["cohort"] = {"n_subjects": int(n_subj), "n_cell_types": int(n_ct),
                         "n_folds": int(args.n_folds)}
    summary["provenance"] = {
        "git_commit": git_sha(_WORKTREE_ROOT),
        "config_path": str(args.config),
        "pred_root": str(args.pred_root),
        "splits_path": str(args.splits_path),
    }
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Wrote %s", summary_path)
    return 0


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="AttnLRP / GMAR / GAF on ResDec-MHE pathology-stratified attention",
    )
    p.add_argument("--config", type=Path,
                   default=Path("configs/resdec_mhe/canonical.yaml"))
    p.add_argument("--pred-root", type=Path, required=True,
                   help="Per-fold output root (canonical seed42 default).")
    p.add_argument("--splits-path", type=Path,
                   default=Path("outputs/splits.json"))
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--metadata-path", type=Path, default=None,
                   help="Override cfg.data.metadata_path (variant pipelines).")
    p.add_argument("--precomputed-dir", type=Path, default=None,
                   help="Override cfg.data.precomputed_dir.")
    return p


if __name__ == "__main__":
    sys.exit(main(_build_argparser().parse_args()))
