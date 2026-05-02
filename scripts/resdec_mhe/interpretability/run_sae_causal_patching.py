"""Orlov 2026 §4.2 SAE causal feature patching for ResDec-MHE canonical.

The 60-run SAE sweep (EXP-020) flagged exactly **1 / 323** SAE features in the
canonical batch_topk/fused/exp32_k64_seed0 config that has Splatter as its
top decoder-direction cell type (feature_idx=572, ct_dominance=0.1546). The
180-run smaller-m sweep (EXP-024-stepE) returned **0 / 180** Splatter-dominant
features under the same criteria — strengthening the distributed-representation
claim.

This script is the causal follow-up: does **patching feature 572 at the SAE
bottleneck** shift downstream cognition predictions in a Splatter-magnitude-
dependent direction?

The canonical SAE was trained on the **fused** layer (per-CT embeddings, shape
``[B, 31, 64]``). Patching pipeline per subject:

    encoder(.) -> fused [B, 31, 64] (via return_embeddings=True)
        -> SAE.encode (per-(B,CT) row of length 64) -> codes [B, 31, 2048]
            -> set codes[:, :, feature_idx] := patch_value
        -> SAE.decode -> patched_fused [B, 31, 64]
    -> pathology_attention(patched_fused, path_emb, mask)
        -> patched_attended [B, 64]
    -> head(patched_attended, metadata)
        -> patched residual (Σ stage_k)
    + y_tabpfn -> patched composite prediction

Three patch modes are evaluated against a no-patch baseline:

  * ``zero``       — patch_value = 0 (knock-out)
  * ``saturate``   — patch_value = 99th percentile of feature 572 across
                     the entire 5-fold per-(subject, CT) cohort
  * ``push_down``  — patch_value = 1st percentile (over nonzero values)

To establish causal **specificity**, the same protocol is run against
``--n-random`` random feature controls (uniformly sampled feature indices
that are NOT feature 572 and NOT dead at the cohort level).

Outputs
-------

  * ``<out-dir>/sae_causal_patching.json`` — per-fold per-mode R², ΔR² vs
    baseline, per-subject Δŷ, Spearman ρ(Δŷ, Splatter cell count), and the
    matching arrays for K random-feature controls.
  * ``<out-dir>/sae_causal_patching.md`` — human-readable summary.
  * ``<out-dir>/figures/sae_causal_patching/fig_sae_causal_patching.{png,pdf}``
    — 4-panel figure (A: Δŷ histogram per mode; B: Δŷ vs Splatter cell count
    scatter; C: ΔR² Splatter-feat vs random-feat distribution; D: per-mode R²
    box plot across 5 folds).

Usage
-----

    cd <worktree-root>
    PYTHONPATH=. \\
    CUDA_VISIBLE_DEVICES=0 \\
    uv run python scripts/resdec_mhe/interpretability/run_sae_causal_patching.py
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import fields
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from scipy.stats import spearmanr
from sklearn.metrics import r2_score

_WORKTREE_ROOT = Path(__file__).resolve().parents[3]
if str(_WORKTREE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKTREE_ROOT))

from src.analysis.sparse_autoencoder import (  # noqa: E402
    SAEConfig,
    SAEModel,
    _decode_numpy,
    _encode_numpy,
)
from src.data.constants import CELL_TYPE_ORDER  # noqa: E402
from src.data.datamodule import CognitiveResilienceDataModule  # noqa: E402
from src.data.splits import load_splits  # noqa: E402
from src.training.resdec_lightning_module import ResDecLightningModule  # noqa: E402
from src.utils.provenance import git_sha, pick_max_r2_ckpt  # noqa: E402

logger = logging.getLogger(__name__)


SPLATTER_CT_NAME = "Splatter"
DEFAULT_SAE_CONFIG_DIR = (
    "outputs/canonical/sae/batch_topk/fused/exp32_k64_seed0"
)
DEFAULT_FUSED_ACTIVATIONS = (
    "outputs/canonical/sae/activations_fused_all_folds.npz"
)


# ─────────────────────────────────────────────────────────────────────────────
# SAE loading
# ─────────────────────────────────────────────────────────────────────────────

def load_sae_from_dir(sae_dir: Path) -> SAEModel:
    """Load a trained SAE from ``<sae_dir>/sae_model.npz``.

    Filters legacy ``l1_lambda`` config keys not present on the current
    SAEConfig dataclass; mirrors :class:`run_sae_random_null` config-match
    semantics.
    """
    sae_npz = sae_dir / "sae_model.npz"
    if not sae_npz.exists():
        raise FileNotFoundError(f"SAE checkpoint missing: {sae_npz}")
    data = np.load(sae_npz, allow_pickle=True)
    cfg_raw = json.loads(str(data["config_json"]))
    valid_keys = {f.name for f in fields(SAEConfig)}
    cfg_dict = {k: v for k, v in cfg_raw.items() if k in valid_keys}
    cfg = SAEConfig(**cfg_dict)
    activation_stats: dict[str, np.ndarray] = {
        "mean": np.asarray(data["stat_mean"]),
        "std": np.asarray(data["stat_std"]),
        "fraction_active": np.asarray(data["stat_fraction_active"]),
        "is_dead": np.asarray(data["stat_is_dead"], dtype=bool),
    }
    if cfg.architecture == "batch_topk":
        activation_stats["threshold"] = np.asarray(data["stat_threshold"])
    return SAEModel(
        W_enc=np.asarray(data["W_enc"], dtype=np.float32),
        b_enc=np.asarray(data["b_enc"], dtype=np.float32),
        W_dec=np.asarray(data["W_dec"], dtype=np.float32),
        b_dec=np.asarray(data["b_dec"], dtype=np.float32),
        config=cfg,
        activation_stats=activation_stats,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Patching primitive
# ─────────────────────────────────────────────────────────────────────────────


def patch_fused_with_sae(
    fused: torch.Tensor,
    sae: SAEModel,
    *,
    feature_idx: int | None,
    patch_value: float | None = None,
) -> torch.Tensor:
    """Apply SAE encode -> optionally patch one feature -> decode to fused.

    Parameters
    ----------
    fused
        ``[B, C, n]`` per-(subject, CT) embedding tensor on any device. Will be
        moved through CPU+numpy for the SAE forward (the SAE is a pure-numpy
        container).
    sae
        Trained SAE (canonical batch_topk/fused config).
    feature_idx
        Index in ``[0, m)`` of the SAE feature to patch. If ``None``, the SAE
        round-trip is performed without patching (used as the "SAE-baseline"
        forward; isolates patch effect from the SAE reconstruction error,
        which is small but non-zero given FVE≈0.999).
    patch_value
        Scalar value to write into the chosen column AFTER encoding. Required
        when ``feature_idx`` is not None.

    Returns
    -------
    torch.Tensor
        Patched (or round-tripped) fused tensor, same shape / dtype / device
        as input.
    """
    if fused.dim() != 3:
        raise ValueError(f"Expected fused [B, C, n] 3D; got {fused.shape}")
    B, C, n = fused.shape
    flat_np = fused.detach().cpu().numpy().reshape(B * C, n)
    h = _encode_numpy(sae, flat_np)  # [B*C, m]
    if feature_idx is not None:
        if patch_value is None:
            raise ValueError(
                "patch_value is required when feature_idx is not None"
            )
        h[:, int(feature_idx)] = float(patch_value)
    x_hat = _decode_numpy(sae, h)  # [B*C, n]
    out = (
        torch.from_numpy(x_hat.astype(np.float32))
        .to(fused.device, dtype=fused.dtype)
        .view(B, C, n)
    )
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Forward through patched fused -> head -> composite prediction
# ─────────────────────────────────────────────────────────────────────────────


def _resolve_d_metadata(model: ResDecLightningModule) -> int:
    """Return the FiLM metadata dimension expected by ``model.head``."""
    return int(model._d_metadata)


def _zero_metadata(B: int, device: torch.device, d_metadata: int) -> torch.Tensor:
    """Canonical zero-metadata vector (FiLM identity) for inference."""
    return torch.zeros(B, d_metadata, device=device)


def compute_path_emb_for_batch(
    model: ResDecLightningModule, batch_d: dict,
) -> torch.Tensor:
    """Recompute the pathology embedding for one batch.

    The encoder's :meth:`forward(..., return_embeddings=True)` returns fused
    and attended in its embeddings dict but NOT path_emb (verified at
    full_model.py:692-698). path_emb is needed by pathology_attention; we
    recompute it from the same region_handler + pathology_encoder path the
    canonical forward uses. Eval-mode + no-dropout, so the result is
    bit-identical to the path_emb consumed inside the canonical forward.
    """
    encoder = model.encoder
    region_pseudobulk = batch_d.get("region_pseudobulk")
    region_mask = batch_d.get("region_mask")
    pseudobulk = batch_d.get("pseudobulk")
    pathology = batch_d.get("pathology")
    if pathology is None:
        raise ValueError("batch missing pathology tensor")
    if region_pseudobulk is None and pseudobulk is not None:
        B = pseudobulk.size(0)
        device = pseudobulk.device
        from src.data.constants import PFC_REGION_IDX
        region_pseudobulk = torch.zeros(
            B, encoder.n_regions, encoder.n_cell_types, encoder.n_genes,
            device=device, dtype=pseudobulk.dtype,
        )
        region_pseudobulk[:, PFC_REGION_IDX, :, :] = pseudobulk
        region_mask = torch.zeros(B, encoder.n_regions, dtype=torch.bool, device=device)
        region_mask[:, PFC_REGION_IDX] = True
    if region_pseudobulk is None:
        raise ValueError("batch lacks both region_pseudobulk and pseudobulk")
    if region_mask is None:
        B = region_pseudobulk.size(0)
        device = region_pseudobulk.device
        region_mask = torch.ones(B, encoder.n_regions, dtype=torch.bool, device=device)

    region_encoded = encoder._encode_hgt_input_per_region(region_pseudobulk, region_mask)
    _, region_context, _ = encoder.region_handler(region_encoded, region_mask)
    return encoder.pathology_encoder(pathology, region_context)


def predict_from_fused(
    model: ResDecLightningModule,
    fused_used: torch.Tensor,
    path_emb: torch.Tensor,
    cell_type_mask: torch.Tensor | None,
) -> np.ndarray:
    """Forward fused -> pathology_attention -> head and return [B] residual."""
    encoder = model.encoder
    if encoder.use_pathology_attention:
        attended, _ = encoder.pathology_attention(
            fused_used, path_emb, cell_type_mask=cell_type_mask,
            return_attention_weights=False,
        )
    else:
        if cell_type_mask is not None:
            mask = cell_type_mask.unsqueeze(-1).float()
            attended = (fused_used * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            attended = fused_used.mean(dim=1)

    B = attended.size(0)
    d_metadata = _resolve_d_metadata(model)
    metadata = _zero_metadata(B, attended.device, d_metadata)
    head_out = model.head(attended, metadata)
    return head_out["prediction"].detach().cpu().numpy().reshape(-1)


def forward_with_patched_fused(
    model: ResDecLightningModule,
    batch_d: dict,
    fused_baseline: torch.Tensor,
    path_emb: torch.Tensor,
    sae: SAEModel,
    *,
    mode: str,
    feature_idx: int | None,
    patch_value: float | None,
) -> np.ndarray:
    """Run pathology_attention + head with optionally-patched fused.

    Parameters
    ----------
    model
        Loaded ResDecLightningModule on device.
    batch_d
        Batched dict (already on device) with ``pathology``, ``cell_type_mask``,
        ``subject_ids``, etc.
    fused_baseline
        Pre-computed ``fused [B, 31, 64]`` from the no-patch encoder forward.
    path_emb
        Pre-computed ``path_emb [B, d_cond]`` (see
        :func:`compute_path_emb_for_batch`).
    sae
        Trained SAE (canonical batch_topk/fused).
    mode
        One of:
          * ``"encoder_baseline"`` — use raw fused (no SAE); recovers canonical R².
          * ``"sae_baseline"`` — SAE encode→decode round-trip (no patch).
          * ``"patch"`` — SAE encode → set feature_idx := patch_value → decode.
    feature_idx
        SAE feature index to patch (only consulted when ``mode == "patch"``).
    patch_value
        Scalar patch value (only consulted when ``mode == "patch"``).

    Returns
    -------
    residual_pred : np.ndarray
        ``[B]`` residual prediction (sum of stage scalars; pre-TabPFN).
    """
    if mode == "encoder_baseline":
        fused_used = fused_baseline
    elif mode == "sae_baseline":
        fused_used = patch_fused_with_sae(
            fused_baseline, sae,
            feature_idx=None, patch_value=None,
        )
    elif mode == "patch":
        if feature_idx is None or patch_value is None:
            raise ValueError("patch mode requires feature_idx and patch_value")
        fused_used = patch_fused_with_sae(
            fused_baseline, sae,
            feature_idx=int(feature_idx), patch_value=float(patch_value),
        )
    else:
        raise ValueError(
            f"unknown mode: {mode!r}; expected one of "
            "{'encoder_baseline', 'sae_baseline', 'patch'}"
        )
    return predict_from_fused(
        model, fused_used, path_emb, batch_d.get("cell_type_mask"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# TabPFN map loading
# ─────────────────────────────────────────────────────────────────────────────


def _load_tabpfn_outer_map(tabpfn_dir: Path, fold: int) -> dict[str, float]:
    """Load fold-specific TabPFN outer ``y_tabpfn`` lookup."""
    p = tabpfn_dir / f"tabpfn_outer_fold{fold}.npz"
    if not p.exists():
        raise FileNotFoundError(f"TabPFN outer cache missing: {p}")
    d = np.load(p, allow_pickle=True)
    return {
        str(s): float(v) for s, v in zip(d["val_subject_ids"], d["y_tabpfn"])
    }


# ─────────────────────────────────────────────────────────────────────────────
# Cohort-level activation statistics for percentile cutoffs
# ─────────────────────────────────────────────────────────────────────────────


def compute_feature_percentiles(
    fused_activations_npz: Path,
    sae: SAEModel,
    *,
    feature_idx: int,
    pct_low: float = 1.0,
    pct_high: float = 99.0,
) -> dict[str, float]:
    """Compute the 1st / 99th percentile of one SAE feature across the cohort.

    Loaded from the pre-extracted ``activations_fused_all_folds.npz`` (5-fold
    pooled cohort, shape ``[N≈2556, 31, 64]``) so the percentile cutoffs are
    grounded in the empirical activation distribution of the canonical
    pipeline. Percentiles are taken over the **full row distribution including
    zeros** (canonical SAE inference; matches what the patched forward will
    reconstruct).
    """
    d = np.load(fused_activations_npz, allow_pickle=True)
    acts = d["activations"]  # [N, 31, 64]
    if acts.ndim != 3:
        raise ValueError(
            f"Expected fused activations 3D [N, 31, 64]; got {acts.shape}"
        )
    flat = acts.reshape(-1, acts.shape[-1])
    h = _encode_numpy(sae, flat)  # [N*31, m]
    col = h[:, int(feature_idx)]
    return {
        f"p{pct_low:g}": float(np.percentile(col, pct_low)),
        f"p{pct_high:g}": float(np.percentile(col, pct_high)),
        "max": float(col.max()),
        "fraction_active": float((col > 0).mean()),
        "n_active": int((col > 0).sum()),
        "n_total": int(len(col)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Random control feature selection
# ─────────────────────────────────────────────────────────────────────────────


def select_random_controls(
    sae: SAEModel,
    *,
    target_feature_idx: int,
    n_random: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample ``n_random`` distinct feature indices that are non-dead and
    not the target feature.

    Drawn uniformly without replacement from
    ``{j : j != target_feature_idx, not is_dead[j]}``.
    """
    is_dead = sae.activation_stats.get("is_dead")
    if is_dead is None:
        eligible = np.arange(sae.W_enc.shape[0])
    else:
        eligible = np.where(~np.asarray(is_dead, dtype=bool))[0]
    eligible = eligible[eligible != int(target_feature_idx)]
    if len(eligible) < n_random:
        raise ValueError(
            f"Not enough non-dead features ({len(eligible)}) for "
            f"n_random={n_random}"
        )
    return rng.choice(eligible, size=int(n_random), replace=False).astype(int)


# ─────────────────────────────────────────────────────────────────────────────
# Per-fold patching loop
# ─────────────────────────────────────────────────────────────────────────────


def precompute_batch_caches(
    model: ResDecLightningModule,
    val_batches: list[dict],
    splatter_ct_idx: int,
    device: torch.device,
) -> list[dict]:
    """Encode all val batches once and cache fused + path_emb + meta.

    Returns one dict per batch with:
      * ``batch_d`` — device-resident batch dict (same content as
        ``val_batches[i]`` but moved to ``device``).
      * ``fused`` — ``[B, 31, 64]`` baseline fused tensor (no SAE).
      * ``path_emb`` — ``[B, d_cond]`` pathology embedding.
      * ``subject_ids``, ``y_true``, ``splatter_cells`` — per-subject metadata
        used for downstream R² / Spearman computation.

    This avoids re-running the encoder once per patch mode (15+ runs per fold
    vs the canonical 1).
    """
    caches: list[dict] = []
    enc_keys = (
        "region_pseudobulk", "region_mask", "pseudobulk",
        "ccc_edge_index", "ccc_edge_type", "ccc_edge_attr",
        "cell_data", "cell_offsets", "cell_type_mask",
        "pathology",
    )
    with torch.no_grad():
        for batch in val_batches:
            batch_d = {
                k: (v.to(device) if torch.is_tensor(v) else v)
                for k, v in batch.items()
            }
            enc_out = model.encoder(
                **{k: batch_d.get(k) for k in enc_keys},
                return_embeddings=True,
            )
            fused = enc_out["embeddings"]["fused"]  # [B, 31, 64]
            path_emb = compute_path_emb_for_batch(model, batch_d)

            sids = [str(s) for s in batch["subject_ids"]]
            cog = batch_d.get("cognition")
            if cog is not None:
                y_true = cog.detach().cpu().numpy().reshape(-1)
            else:
                y_true = np.full(len(sids), np.nan, dtype=np.float64)
            cell_counts = batch.get("cell_counts")
            if cell_counts is not None and torch.is_tensor(cell_counts):
                splatter_cells = (
                    cell_counts[:, splatter_ct_idx].detach().cpu().numpy().astype(np.int64)
                )
            else:
                splatter_cells = np.full(len(sids), -1, dtype=np.int64)

            caches.append({
                "batch_d": batch_d,
                "fused": fused,
                "path_emb": path_emb,
                "subject_ids": sids,
                "y_true": y_true,
                "splatter_cells": splatter_cells,
            })
    return caches


def _gather_val_outputs(
    model: ResDecLightningModule,
    cached_batches: list[dict],
    sae: SAEModel,
    *,
    mode: str,
    feature_idx: int | None,
    patch_value: float | None,
    tabpfn_map: dict[str, float],
) -> dict[str, np.ndarray]:
    """Run val forward (optionally patched) using precomputed batch caches."""
    sids_all: list[str] = []
    composites: list[float] = []
    truths: list[float] = []
    splatter_counts: list[int] = []
    with torch.no_grad():
        for cache in cached_batches:
            residuals = forward_with_patched_fused(
                model, cache["batch_d"], cache["fused"], cache["path_emb"], sae,
                mode=mode, feature_idx=feature_idx, patch_value=patch_value,
            )
            for i, sid in enumerate(cache["subject_ids"]):
                ytab = tabpfn_map.get(sid, np.nan)
                composites.append(float(residuals[i]) + float(ytab))
                truths.append(float(cache["y_true"][i]))
                splatter_counts.append(int(cache["splatter_cells"][i]))
                sids_all.append(sid)

    return {
        "subject_ids": np.asarray(sids_all, dtype=object),
        "composite": np.asarray(composites, dtype=np.float64),
        "truth": np.asarray(truths, dtype=np.float64),
        "splatter_cells": np.asarray(splatter_counts, dtype=np.int64),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "--config",
        default="configs/resdec_mhe/canonical.yaml",
        help="Canonical model config; merged on top of configs/default.yaml.",
    )
    p.add_argument(
        "--canonical-dir",
        default="outputs/canonical/p5_canonical_seed42",
        help="Per-fold checkpoint root (best-*.ckpt resolved by max val R²).",
    )
    p.add_argument("--tabpfn-dir", default="data/canonical")
    p.add_argument("--splits-path", default="outputs/splits.json")
    p.add_argument(
        "--sae-dir",
        default=DEFAULT_SAE_CONFIG_DIR,
        help="Trained-SAE config directory (containing sae_model.npz).",
    )
    p.add_argument(
        "--fused-activations-npz",
        default=DEFAULT_FUSED_ACTIVATIONS,
        help="Pooled-fold fused activations for percentile cutoff calculation.",
    )
    p.add_argument("--feature-idx", type=int, default=572)
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--n-random", type=int, default=10)
    p.add_argument("--random-seed", type=int, default=42)
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--out-dir",
        default="outputs/canonical/interpretability",
        help="Root for sae_causal_patching.{json,md} + figure subdir.",
    )
    p.add_argument(
        "--no-figure", action="store_true",
        help="Skip figure rendering (still emits JSON + MD).",
    )
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures" / "sae_causal_patching"
    fig_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    sae_dir = Path(args.sae_dir)
    sae = load_sae_from_dir(sae_dir)
    cfg_dict_for_log = sae.config
    logger.info(
        "Loaded SAE: arch=%s expansion=%s k=%s seed=%s from %s",
        cfg_dict_for_log.architecture, cfg_dict_for_log.expansion,
        cfg_dict_for_log.k, cfg_dict_for_log.seed, sae_dir,
    )

    # Resolve Splatter CT index from canonical CT order.
    splatter_ct_idx = list(CELL_TYPE_ORDER).index(SPLATTER_CT_NAME)

    # Cohort-level percentile cutoffs.
    fused_npz = Path(args.fused_activations_npz)
    pct_stats = compute_feature_percentiles(
        fused_npz, sae, feature_idx=args.feature_idx,
        pct_low=1.0, pct_high=99.0,
    )
    logger.info(
        "Feature %d cohort stats: %s", args.feature_idx, json.dumps(pct_stats),
    )
    p1 = pct_stats["p1"]
    p99 = pct_stats["p99"]

    # Random-feature controls.
    rng = np.random.default_rng(args.random_seed)
    random_feature_idxs = select_random_controls(
        sae, target_feature_idx=args.feature_idx, n_random=args.n_random, rng=rng,
    ).tolist()
    logger.info(
        "Selected %d random control features: %s",
        args.n_random, random_feature_idxs,
    )

    # Per-feature patch modes for the canonical Splatter feature.
    # Two baselines: "encoder_baseline" (raw fused, recovers canonical R²) and
    # "sae_baseline" (SAE encode→decode round-trip, no patch — isolates the
    # patch effect from the SAE's small reconstruction error). ΔR² is reported
    # against the sae_baseline so the patch perturbation is the only difference
    # between baseline and patched forward passes.
    splatter_patch_values: dict[str, float] = {
        "zero": 0.0,
        "saturate": p99,
        "push_down": p1,
    }
    # Random controls only run the "saturate" patch — that's the most
    # information-bearing single perturbation under the same protocol; running
    # 3 modes × 10 features × 5 folds × ~100 val subjects becomes expensive
    # without adding statistical specificity beyond the saturate ΔR² baseline.
    random_mode_value = p99

    # Per-fold accumulators.
    fold_results: list[dict] = []
    cfg_full = OmegaConf.merge(
        OmegaConf.load(_WORKTREE_ROOT / "configs" / "default.yaml"),
        OmegaConf.load(_WORKTREE_ROOT / args.config),
    )
    OmegaConf.set_struct(cfg_full, False)
    cfg_full.model.head.type = "deterministic"

    t_start = time.time()
    for fold in range(args.n_folds):
        fold_cfg = OmegaConf.create(OmegaConf.to_container(cfg_full, resolve=True))
        OmegaConf.set_struct(fold_cfg, False)
        fold_cfg.data.fold = int(fold)

        fold_dir = Path(args.canonical_dir) / f"fold{fold}"
        ckpt_path = pick_max_r2_ckpt(fold_dir / "checkpoints")
        logger.info("fold %d: loading %s", fold, ckpt_path.name)

        splits = load_splits(str(args.splits_path))
        meta_path = Path(fold_cfg.data.metadata_path) / "metadata.csv"
        if not meta_path.is_absolute():
            meta_path = _WORKTREE_ROOT / meta_path
        metadata_df = pd.read_csv(meta_path)
        dm = CognitiveResilienceDataModule(
            config=fold_cfg, metadata=metadata_df, splits=splits,
            fold_idx=int(fold),
            precomputed_dir=fold_cfg.data.precomputed_dir,
            adata=None,
        )
        dm.setup(stage="fit")

        model = ResDecLightningModule.load_from_checkpoint(
            str(ckpt_path), config=fold_cfg, map_location="cpu",
        ).to(device).eval()

        val_batches: list[dict] = list(dm.val_dataloader())
        tabpfn_map = _load_tabpfn_outer_map(
            Path(args.tabpfn_dir), int(fold),
        )

        # Precompute fused + path_emb once per batch (single encoder forward
        # per batch shared across 15+ patch modes below).
        cached_batches = precompute_batch_caches(
            model, val_batches, splatter_ct_idx, device,
        )

        # 1a) Encoder baseline (raw fused, no SAE) — recovers canonical R².
        enc_baseline_out = _gather_val_outputs(
            model, cached_batches, sae,
            mode="encoder_baseline", feature_idx=None, patch_value=None,
            tabpfn_map=tabpfn_map,
        )
        enc_baseline_r2 = float(r2_score(
            enc_baseline_out["truth"], enc_baseline_out["composite"],
        ))
        logger.info(
            "fold %d: encoder_baseline R²=%+.4f n=%d",
            fold, enc_baseline_r2, len(enc_baseline_out["truth"]),
        )

        # 1b) SAE baseline (round-trip, no patch) — isolates patch effect.
        sae_baseline_out = _gather_val_outputs(
            model, cached_batches, sae,
            mode="sae_baseline", feature_idx=None, patch_value=None,
            tabpfn_map=tabpfn_map,
        )
        sae_baseline_r2 = float(r2_score(
            sae_baseline_out["truth"], sae_baseline_out["composite"],
        ))
        logger.info(
            "fold %d: sae_baseline R²=%+.4f (vs canonical %+.4f, Δ_round_trip=%+.4f)",
            fold, sae_baseline_r2, enc_baseline_r2,
            sae_baseline_r2 - enc_baseline_r2,
        )

        # 1c) Splatter feature: 3 patch modes (zero, saturate, push_down).
        splatter_patch_outputs: dict[str, dict[str, np.ndarray]] = {}
        for mode_name, patch_val in splatter_patch_values.items():
            res = _gather_val_outputs(
                model, cached_batches, sae,
                mode="patch", feature_idx=args.feature_idx, patch_value=patch_val,
                tabpfn_map=tabpfn_map,
            )
            splatter_patch_outputs[mode_name] = res
            r2 = float(r2_score(res["truth"], res["composite"]))
            logger.info(
                "fold %d: Splatter patch=%-9s R²=%+.4f ΔR²=%+.4f",
                fold, mode_name, r2, r2 - sae_baseline_r2,
            )

        # 2) Random feature controls: only "saturate" mode at p99.
        random_outputs: dict[int, dict[str, np.ndarray]] = {}
        for ridx in random_feature_idxs:
            res = _gather_val_outputs(
                model, cached_batches, sae,
                mode="patch", feature_idx=int(ridx), patch_value=random_mode_value,
                tabpfn_map=tabpfn_map,
            )
            random_outputs[int(ridx)] = res

        # Compute per-fold per-mode metrics.
        per_mode = {}
        for mode_name in ("zero", "saturate", "push_down"):
            mode_r2 = float(r2_score(
                splatter_patch_outputs[mode_name]["truth"],
                splatter_patch_outputs[mode_name]["composite"],
            ))
            d_yhat = (
                splatter_patch_outputs[mode_name]["composite"]
                - sae_baseline_out["composite"]
            )
            splatter_cells = splatter_patch_outputs[mode_name]["splatter_cells"]
            valid_cells = splatter_cells != -1
            if valid_cells.sum() >= 5 and np.std(splatter_cells[valid_cells]) > 0 \
                    and np.std(d_yhat[valid_cells]) > 0:
                rho_val, p_val = spearmanr(
                    d_yhat[valid_cells], splatter_cells[valid_cells],
                )
                rho = float(rho_val)
                p_rho = float(p_val)
            else:
                rho = float("nan")
                p_rho = float("nan")
            per_mode[mode_name] = {
                "r2": mode_r2,
                "delta_r2_vs_sae_baseline": mode_r2 - sae_baseline_r2,
                "delta_r2_vs_encoder_baseline": mode_r2 - enc_baseline_r2,
                "delta_yhat_mean": float(d_yhat.mean()),
                "delta_yhat_std": float(d_yhat.std(ddof=1)) if len(d_yhat) > 1 else 0.0,
                "delta_yhat": d_yhat.tolist(),
                "spearman_rho_dyhat_splatter_cells": rho,
                "spearman_p": p_rho,
            }

        random_per_feature = {}
        for ridx in random_feature_idxs:
            mode_r2 = float(r2_score(
                random_outputs[int(ridx)]["truth"],
                random_outputs[int(ridx)]["composite"],
            ))
            d_yhat_random = (
                random_outputs[int(ridx)]["composite"]
                - sae_baseline_out["composite"]
            )
            random_per_feature[int(ridx)] = {
                "r2": mode_r2,
                "delta_r2_vs_sae_baseline": mode_r2 - sae_baseline_r2,
                "delta_r2_vs_encoder_baseline": mode_r2 - enc_baseline_r2,
                "delta_yhat_mean": float(d_yhat_random.mean()),
                "delta_yhat_std": float(d_yhat_random.std(ddof=1))
                    if len(d_yhat_random) > 1 else 0.0,
            }

        fold_results.append({
            "fold": int(fold),
            "n_val": int(len(enc_baseline_out["truth"])),
            "encoder_baseline_r2": enc_baseline_r2,
            "sae_baseline_r2": sae_baseline_r2,
            "splatter_feature": {
                "feature_idx": int(args.feature_idx),
                "per_mode": per_mode,
                "subject_ids": enc_baseline_out["subject_ids"].tolist(),
                "splatter_cells": enc_baseline_out["splatter_cells"].tolist(),
                "y_truth": enc_baseline_out["truth"].tolist(),
                "y_encoder_baseline": enc_baseline_out["composite"].tolist(),
                "y_sae_baseline": sae_baseline_out["composite"].tolist(),
            },
            "random_feature_controls": {
                "patch_mode": "saturate",
                "patch_value": float(random_mode_value),
                "feature_idxs": random_feature_idxs,
                "per_feature": random_per_feature,
            },
        })
        del cached_batches, val_batches, model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    elapsed_min = (time.time() - t_start) / 60.0

    # Aggregate across folds.
    encoder_per_fold = [fr["encoder_baseline_r2"] for fr in fold_results]
    sae_per_fold = [fr["sae_baseline_r2"] for fr in fold_results]
    encoder_mean_r2 = float(np.mean(encoder_per_fold))
    sae_mean_r2 = float(np.mean(sae_per_fold))

    splatter_aggregate: dict[str, dict] = {}
    for mode_name in ("zero", "saturate", "push_down"):
        delta_r2_per_fold = [
            fr["splatter_feature"]["per_mode"][mode_name]["delta_r2_vs_sae_baseline"]
            for fr in fold_results
        ]
        rho_per_fold = [
            fr["splatter_feature"]["per_mode"][mode_name][
                "spearman_rho_dyhat_splatter_cells"
            ]
            for fr in fold_results
        ]
        # nanmean over all-NaN list emits RuntimeWarning; guard explicitly.
        finite_rho = [r for r in rho_per_fold if r is not None and not np.isnan(r)]
        splatter_aggregate[mode_name] = {
            "delta_r2_per_fold": delta_r2_per_fold,
            "delta_r2_mean": float(np.mean(delta_r2_per_fold)),
            "delta_r2_std": float(np.std(delta_r2_per_fold, ddof=1))
                if len(delta_r2_per_fold) > 1 else 0.0,
            "spearman_rho_per_fold": rho_per_fold,
            "spearman_rho_mean": float(np.mean(finite_rho))
                if len(finite_rho) > 0 else float("nan"),
        }

    random_aggregate: dict[str, list[float]] = {}
    for ridx in random_feature_idxs:
        delta_r2_per_fold = [
            fr["random_feature_controls"]["per_feature"][int(ridx)][
                "delta_r2_vs_sae_baseline"
            ]
            for fr in fold_results
        ]
        random_aggregate[str(ridx)] = delta_r2_per_fold

    flat_random_delta_r2 = [
        v for lst in random_aggregate.values() for v in lst
    ]
    splatter_saturate_delta_r2 = splatter_aggregate["saturate"]["delta_r2_per_fold"]

    summary = {
        "experiment": "EXP-042-sae-causal-patching",
        "method": "Orlov 2026 §4.2 SAE causal feature patching",
        "sae_config": {
            "dir": str(sae_dir),
            "architecture": cfg_dict_for_log.architecture,
            "expansion": int(cfg_dict_for_log.expansion),
            "k": int(cfg_dict_for_log.k) if cfg_dict_for_log.k is not None else None,
            "seed": int(cfg_dict_for_log.seed),
            "layer": "fused",
        },
        "feature_idx": int(args.feature_idx),
        "feature_cohort_stats": pct_stats,
        "patch_modes": {
            "zero": 0.0,
            "saturate": float(p99),
            "push_down": float(p1),
        },
        "n_folds": int(args.n_folds),
        "n_random_controls": int(args.n_random),
        "random_seed": int(args.random_seed),
        "encoder_baseline_per_fold_r2": encoder_per_fold,
        "encoder_baseline_mean_r2": encoder_mean_r2,
        "sae_baseline_per_fold_r2": sae_per_fold,
        "sae_baseline_mean_r2": sae_mean_r2,
        "splatter_feature_aggregate": splatter_aggregate,
        "random_feature_aggregate": random_aggregate,
        "summary_statistics": {
            "splatter_saturate_delta_r2_mean": float(np.mean(splatter_saturate_delta_r2)),
            "splatter_saturate_delta_r2_std": float(np.std(splatter_saturate_delta_r2, ddof=1))
                if len(splatter_saturate_delta_r2) > 1 else 0.0,
            "random_saturate_delta_r2_mean": float(np.mean(flat_random_delta_r2)),
            "random_saturate_delta_r2_std": float(np.std(flat_random_delta_r2, ddof=1))
                if len(flat_random_delta_r2) > 1 else 0.0,
            "n_random_pooled": int(len(flat_random_delta_r2)),
        },
        "per_fold": fold_results,
        "provenance": {
            "elapsed_min": round(elapsed_min, 2),
            "device": str(device),
            "git_commit": git_sha(_WORKTREE_ROOT),
        },
    }

    # JSON output.
    json_path = out_dir / "sae_causal_patching.json"
    json_path.write_text(json.dumps(summary, indent=2, default=str))
    logger.info("wrote %s", json_path)

    # Markdown output.
    md_path = out_dir / "sae_causal_patching.md"
    md_path.write_text(_render_markdown(summary))
    logger.info("wrote %s", md_path)

    if not args.no_figure:
        fig_path_stem = fig_dir / "fig_sae_causal_patching"
        try:
            _render_figure(summary, fig_path_stem)
        except Exception as exc:  # pragma: no cover - figure render is best-effort
            logger.warning("figure render failed: %s", exc)
    logger.info("done in %.1f min", elapsed_min)
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# Markdown rendering
# ─────────────────────────────────────────────────────────────────────────────


def _render_markdown(summary: dict) -> str:
    """Render a human-readable summary of the patching results."""
    lines: list[str] = []
    cfg = summary["sae_config"]
    lines.append("# SAE Causal Patching — Splatter feature 572")
    lines.append("")
    lines.append(
        f"SAE config: **{cfg['architecture']}/{cfg['layer']}/exp{cfg['expansion']}_k{cfg['k']}**"
        f" (seed {cfg['seed']})"
    )
    lines.append(f"Feature index: **{summary['feature_idx']}**")
    fc = summary["feature_cohort_stats"]
    lines.append(
        f"Feature cohort stats: fraction_active={fc['fraction_active']:.4f} "
        f"(n_active={fc['n_active']}/{fc['n_total']}); p1={fc['p1']:.4f}, "
        f"p99={fc['p99']:.4f}, max={fc['max']:.4f}"
    )
    lines.append(f"N folds: {summary['n_folds']}; N random controls: {summary['n_random_controls']}")
    lines.append("")
    lines.append("## Patch values")
    pm = summary["patch_modes"]
    lines.append(f"- zero: {pm['zero']:.4f}")
    lines.append(f"- saturate (p99): {pm['saturate']:.4f}")
    lines.append(f"- push_down (p1): {pm['push_down']:.4f}")
    lines.append("")
    lines.append(f"## Per-fold baselines")
    lines.append(
        f"- encoder_baseline R² (raw fused, no SAE) per-fold: "
        f"{[round(v, 4) for v in summary['encoder_baseline_per_fold_r2']]}"
    )
    lines.append(
        f"- encoder_baseline mean: {summary['encoder_baseline_mean_r2']:.4f}"
    )
    lines.append(
        f"- sae_baseline R² (SAE encode→decode round-trip) per-fold: "
        f"{[round(v, 4) for v in summary['sae_baseline_per_fold_r2']]}"
    )
    lines.append(
        f"- sae_baseline mean: {summary['sae_baseline_mean_r2']:.4f}"
    )
    lines.append("")
    lines.append("## Splatter feature 572 — patch-mode aggregate")
    lines.append("| Mode | ΔR² mean | ΔR² std | Spearman ρ(Δŷ, splatter cells) mean |")
    lines.append("|---|---:|---:|---:|")
    for mode_name in ("zero", "saturate", "push_down"):
        agg = summary["splatter_feature_aggregate"][mode_name]
        lines.append(
            f"| {mode_name} | {agg['delta_r2_mean']:+.4f} | {agg['delta_r2_std']:.4f} "
            f"| {agg['spearman_rho_mean']:+.4f} |"
        )
    lines.append("")
    lines.append("## Random feature controls (saturate mode)")
    rs = summary["summary_statistics"]
    lines.append(
        f"- pooled mean ΔR² (across {rs['n_random_pooled']} fold-feature pairs): "
        f"{rs['random_saturate_delta_r2_mean']:+.4f} "
        f"± {rs['random_saturate_delta_r2_std']:.4f}"
    )
    lines.append(
        f"- Splatter saturate ΔR² mean (5-fold): "
        f"{rs['splatter_saturate_delta_r2_mean']:+.4f} "
        f"± {rs['splatter_saturate_delta_r2_std']:.4f}"
    )
    lines.append("")
    lines.append("## Interpretation")
    splatter_eff = abs(rs["splatter_saturate_delta_r2_mean"])
    random_eff = abs(rs["random_saturate_delta_r2_mean"])
    random_std = rs["random_saturate_delta_r2_std"]
    canonical_r2 = abs(summary["encoder_baseline_mean_r2"])
    relative_effect = (
        splatter_eff / canonical_r2 if canonical_r2 > 0 else float("nan")
    )
    # Three-criterion test: (a) splatter Δ exceeds 2× random control mean,
    # (b) splatter Δ exceeds 1 SD of random distribution, (c) absolute
    # magnitude ≥ 1 % of the canonical R² (so we don't claim causation
    # over noise-floor effects that round to zero in the manuscript).
    rel_threshold = 0.01
    cond_a = splatter_eff > 2 * random_eff
    cond_b = splatter_eff > random_std
    cond_c = relative_effect >= rel_threshold
    lines.append(
        f"- |Splatter ΔR²| = {splatter_eff:.4f}; |random ΔR²| (mean) = "
        f"{random_eff:.4f}; random std = {random_std:.4f}; relative effect = "
        f"{relative_effect * 100:.2f} % of canonical R² = {canonical_r2:.4f}"
    )
    lines.append(
        f"- Criterion A (Δ > 2× random mean): {'PASS' if cond_a else 'FAIL'}; "
        f"Criterion B (Δ > random SD): {'PASS' if cond_b else 'FAIL'}; "
        f"Criterion C (Δ ≥ 1 % canonical R²): {'PASS' if cond_c else 'FAIL'}"
    )
    if cond_a and cond_b and cond_c:
        verdict = (
            "**Causal**: the feature changes predictions in a magnitude-meaningful "
            "way that exceeds the random-feature noise floor."
        )
    elif cond_a or cond_b:
        verdict = (
            "**Inconclusive**: the patch effect exceeds at least one specificity "
            "criterion but not the magnitude criterion (or vice versa). The "
            "1/323 Splatter feature is not causally load-bearing at the "
            "manuscript-decision level."
        )
    else:
        verdict = (
            "**Correlated-only, not causal**: the patch effect is no larger "
            "than the random-feature noise floor and is < 1 % of canonical R². "
            "Consistent with §31.11 distributed-representation framing — "
            "the 1/323 Splatter feature is correlated with Splatter at the "
            "decoder direction but does NOT carry the cell-type's predictive "
            "signal in a single-feature-causal sense."
        )
    lines.append(f"- Verdict: {verdict}")
    return "\n".join(lines) + "\n"


# ─────────────────────────────────────────────────────────────────────────────
# Figure rendering
# ─────────────────────────────────────────────────────────────────────────────


def _render_figure(summary: dict, fig_path_stem: Path) -> None:
    """Render the 4-panel figure (saved as PNG + PDF, 600 DPI)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.visualization.theme import apply_theme, fmt_axes, save_fig
    apply_theme(style="paper")

    fold_results = summary["per_fold"]

    # Panel A: per-subject Δŷ histogram across 3 patch modes (overlaid).
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 9.0))
    ax = axes[0, 0]
    mode_colors = {
        "zero": "#1f77b4",
        "saturate": "#d62728",
        "push_down": "#2ca02c",
    }
    for mode_name in ("zero", "saturate", "push_down"):
        all_d = []
        for fr in fold_results:
            all_d.extend(fr["splatter_feature"]["per_mode"][mode_name]["delta_yhat"])
        ax.hist(
            all_d, bins=40, alpha=0.55, color=mode_colors[mode_name],
            label=f"{mode_name} (n={len(all_d)})",
        )
    ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Δŷ (patched − baseline)")
    ax.set_ylabel("Subjects (pooled across folds)")
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    ax.set_title("A. Per-subject Δŷ across 3 patch modes")
    fmt_axes(ax)

    # Panel B: Δŷ vs Splatter cell count scatter (saturate mode).
    ax = axes[0, 1]
    all_d, all_cells = [], []
    for fr in fold_results:
        d = fr["splatter_feature"]["per_mode"]["saturate"]["delta_yhat"]
        cells = fr["splatter_feature"]["splatter_cells"]
        all_d.extend(d)
        all_cells.extend(cells)
    all_d = np.asarray(all_d)
    all_cells = np.asarray(all_cells)
    valid = all_cells != -1
    if valid.sum() >= 5:
        ax.scatter(
            all_cells[valid], all_d[valid], s=14, alpha=0.55,
            color=mode_colors["saturate"], edgecolor="white", linewidth=0.4,
        )
        if all_cells[valid].std() > 0 and all_d[valid].std() > 0:
            from scipy.stats import spearmanr as _sp
            rho_val, p_val = _sp(all_d[valid], all_cells[valid])
            ax.text(
                0.02, 0.98,
                f"Spearman ρ = {float(rho_val):+.3f} (p={float(p_val):.2e})\nn={int(valid.sum())}",
                transform=ax.transAxes, va="top", ha="left", fontsize=9,
            )
    ax.set_xlabel("Splatter cells per subject")
    ax.set_ylabel("Δŷ (saturate − baseline)")
    ax.set_title("B. Δŷ scales with Splatter cell count?")
    ax.set_xscale("symlog", linthresh=1)
    fmt_axes(ax)

    # Panel C: ΔR² Splatter feature vs random features (per fold), saturate mode.
    ax = axes[1, 0]
    splat_d = summary["splatter_feature_aggregate"]["saturate"]["delta_r2_per_fold"]
    rand_d_pooled = []
    for ridx_list in summary["random_feature_aggregate"].values():
        rand_d_pooled.extend(ridx_list)
    n_folds = len(splat_d)
    x_pos = np.arange(n_folds)
    width = 0.35
    ax.bar(
        x_pos - width / 2, splat_d, width, color=mode_colors["saturate"],
        label="Splatter feat 572", edgecolor="black", linewidth=0.4,
    )
    rand_per_fold_mean = []
    for f in range(n_folds):
        per_f = [
            summary["random_feature_aggregate"][rk][f]
            for rk in summary["random_feature_aggregate"].keys()
        ]
        rand_per_fold_mean.append(np.mean(per_f))
    ax.bar(
        x_pos + width / 2, rand_per_fold_mean, width,
        color="#888888", label=f"Random feats (mean of {len(summary['random_feature_aggregate'])})",
        edgecolor="black", linewidth=0.4,
    )
    ax.axhline(0, color="black", linestyle="-", linewidth=0.6)
    ax.set_xlabel("Fold")
    ax.set_ylabel("ΔR² vs canonical")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"F{f}" for f in range(n_folds)])
    ax.legend(loc="best", frameon=False, fontsize=9)
    ax.set_title("C. ΔR² (saturate): Splatter vs random features")
    fmt_axes(ax)

    # Panel D: per-mode R² distribution across 5 folds (box plot).
    ax = axes[1, 1]
    box_data: list[list[float]] = []
    box_labels: list[str] = []
    box_data.append(summary["encoder_baseline_per_fold_r2"])
    box_labels.append("encoder")
    box_data.append(summary["sae_baseline_per_fold_r2"])
    box_labels.append("sae_rt")
    for mode_name in ("zero", "saturate", "push_down"):
        per_fold_r2 = [
            fr["splatter_feature"]["per_mode"][mode_name]["r2"] for fr in fold_results
        ]
        box_data.append(per_fold_r2)
        box_labels.append(mode_name)
    bp = ax.boxplot(
        box_data, labels=box_labels, patch_artist=True, widths=0.6,
        medianprops={"color": "black"},
    )
    box_palette = {
        "encoder": "#bbbbbb",
        "sae_rt": "#888888",
        "zero": mode_colors["zero"],
        "saturate": mode_colors["saturate"],
        "push_down": mode_colors["push_down"],
    }
    for patch, name in zip(bp["boxes"], box_labels):
        patch.set_facecolor(box_palette[name])
        if name not in ("encoder", "sae_rt"):
            patch.set_alpha(0.7)
    ax.axhline(
        summary["encoder_baseline_mean_r2"], color="black", linestyle="--",
        linewidth=0.8,
        label=f"encoder R² = {summary['encoder_baseline_mean_r2']:.4f}",
    )
    ax.set_ylabel("R²")
    ax.set_title("D. Per-mode R² across 5 folds")
    ax.legend(loc="lower left", frameon=False, fontsize=9)
    ax.tick_params(axis="x", labelrotation=15)
    fmt_axes(ax)

    fig.tight_layout()
    save_fig(fig, fig_path_stem, dpi=600, formats=("png",))
    # save_fig is PNG-only by project convention; the task brief explicitly
    # asks for PDF as well, so we emit it directly.
    fig.savefig(
        Path(str(fig_path_stem) + ".pdf"), dpi=600, bbox_inches="tight",
    )


if __name__ == "__main__":
    raise SystemExit(main())
