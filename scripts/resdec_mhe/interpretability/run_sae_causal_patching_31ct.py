"""Symmetric across-CT SAE causal feature patching (31-CT extension).

Companion to ``run_sae_causal_patching.py`` (Splatter-only feature 572). The
canonical experiment showed patching the 1/323 Splatter-dominant SAE feature
shifts R² no more than the random-feature noise floor — supporting the
distributed-representation interpretation. This script generalises that test
SYMMETRICALLY across all 31 cell types so the conclusion is not narrative-
narrow on Splatter alone.

For each of 31 CTs, find the top SAE feature for that CT in the canonical
``batch_topk/fused/exp32_k64_seed0`` SAE:

  1. **Relaxed-pool match**: among the 323 features that pass the relaxed
     filter (non-dead AND ``mw_p_cognition < 0.05`` AND ``fraction_active in
     [1e-4, 0.5]``; ``ct_dominance > 0.7`` clause from ``interpretable_candidate``
     dropped), pick those with ``top_cell_types[0]["cell_type"] == this_CT``
     and choose the feature with the HIGHEST ``ct_dominance`` (most
     concentrated on this CT within the relaxed pool).
  2. **Fallback**: if zero relaxed-pool features have this CT as top-1,
     scan ALL non-dead features in the 2048-feature dictionary and pick the
     feature with maximum decoder-direction projection ``μ_CT @ W_dec[:, j]``
     (the same per-CT decoder-direction logic that ``interpret_features`` uses
     for ``top_cell_types``).

Per fold and per chosen CT-top-feature, run:

    encoder(.) -> fused [B, 31, 64]
        -> SAE.encode -> codes [B*31, 2048]
            -> set codes[:, feature_idx] := p99 cohort value (saturate)
        -> SAE.decode -> patched_fused
    -> pathology_attention -> head -> patched residual
    + y_tabpfn -> patched composite

ΔR² is reported against the ``sae_baseline`` (SAE encode→decode round-trip,
no patch — isolates patch effect from the SAE's small reconstruction error;
matches the existing ``run_sae_causal_patching.py`` convention).

The same 10-feature random-control null is run (saturate at the cohort p99 of
each random feature, exactly mirroring the existing run) for cross-experiment
consistency. Random controls are sampled from non-dead features that are NOT
in the 31-CT chosen set.

Outputs
-------

  * ``<out-dir>/sae_causal_patching_31ct.json`` — schema:
    ``per_ct[].{ct_index, ct_name, feature_idx, feature_source, ct_dominance,
    feature_cohort_stats, per_fold[], saturate_delta_r2_mean, saturate_delta_r2_std}``,
    plus ``random_feature_aggregate``, ``summary_statistics``, ``provenance``.

Usage
-----

    cd <worktree-root>
    PYTHONPATH=. \\
    CUDA_VISIBLE_DEVICES=0 \\
    uv run python scripts/resdec_mhe/interpretability/run_sae_causal_patching_31ct.py
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from sklearn.metrics import r2_score

_WORKTREE_ROOT = Path(__file__).resolve().parents[3]
if str(_WORKTREE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKTREE_ROOT))

from src.analysis.sparse_autoencoder import SAEModel  # noqa: E402
from src.data.constants import CELL_TYPE_ORDER, N_CELL_TYPES  # noqa: E402
from src.data.datamodule import CognitiveResilienceDataModule  # noqa: E402
from src.data.splits import load_splits  # noqa: E402
from src.training.resdec_lightning_module import ResDecLightningModule  # noqa: E402
from src.utils.provenance import git_sha, pick_max_r2_ckpt  # noqa: E402

# Reuse existing patching infrastructure verbatim. Only the outer loop changes.
# select_random_controls is reused with target_feature_idx=572 (the existing
# experiment's Splatter feature) so the 10 random features + patch value
# match the original `sae_causal_patching.json` random_aggregate exactly,
# enabling cross-experiment value verification (-7.3e-5 ± 1.19e-3 at seed=42).
from scripts.resdec_mhe.interpretability.run_sae_causal_patching import (  # noqa: E402
    DEFAULT_FUSED_ACTIVATIONS,
    DEFAULT_SAE_CONFIG_DIR,
    _gather_val_outputs,
    _load_tabpfn_outer_map,
    compute_feature_percentiles,
    load_sae_from_dir,
    precompute_batch_caches,
    select_random_controls,
)
# Splatter feature index used by the existing run_sae_causal_patching.py (the
# 1/323 relaxed-pool feature with Splatter as top-CT). Used here ONLY for
# random-control parity (target_feature_idx exclusion + p99 patch value), so
# the random null reproduces the existing experiment's value exactly.
SPLATTER_FEATURE_IDX_LEGACY: int = 572

logger = logging.getLogger(__name__)


# Lower bound for non-dead activation; matches DEAD_FRACTION_THRESHOLD in
# src/analysis/sparse_autoencoder.py and run_feature_xref_consensus.py.
DEAD_FRACTION_THRESHOLD: float = 1e-4
# Splatter still needed for the cell_counts axis index in precompute_batch_caches
# (which expects a CT to use for the Spearman-correlation per-subject cell count
# field). For this experiment we keep the same axis but do not interpret the
# cell-count value beyond passing it through to the JSON.
SPLATTER_CT_NAME = "Splatter"


# ─────────────────────────────────────────────────────────────────────────────
# Per-CT top-feature selection
# ─────────────────────────────────────────────────────────────────────────────


def _load_feature_report(sae_dir: Path) -> list[dict]:
    """Load ``feature_report.json`` from the SAE config dir."""
    p = sae_dir / "feature_report.json"
    if not p.exists():
        raise FileNotFoundError(f"SAE feature report missing: {p}")
    return json.loads(p.read_text())


def _filter_relaxed(reports: list[dict]) -> list[dict]:
    """Replicate the ``run_feature_xref_consensus._relaxed_filter`` definition.

    Filter spec: non-dead AND ``mw_p_cognition < 0.05`` AND ``fraction_active``
    in ``[1e-4, 0.5]``. The ``ct_dominance > 0.7`` clause from
    ``interpretable_candidate`` is dropped (323 features in the canonical
    fused/exp32_k64_seed0 SAE).
    """
    out: list[dict] = []
    for r in reports:
        if "dead" in r.get("flags", []):
            continue
        p = r.get("mw_p_cognition")
        if p is None or p >= 0.05:
            continue
        f = r.get("fraction_active", 0.0)
        if not (DEAD_FRACTION_THRESHOLD <= f <= 0.5):
            continue
        out.append(r)
    return out


def _compute_per_ct_means(activations_npz: Path) -> tuple[np.ndarray, list[str]]:
    """Load fused activations and return per-CT mean ``[C, n]`` + CT name list.

    The per-CT means are averaged across all subjects (5-fold pooled cohort);
    matches the ``per_ct_means`` precomputed inside
    :func:`src.analysis.sparse_autoencoder.interpret_features` (line 1178)
    used to populate ``top_cell_types``.
    """
    d = np.load(activations_npz, allow_pickle=True)
    acts = d["activations"]  # [N, 31, 64]
    if acts.ndim != 3:
        raise ValueError(
            f"Expected fused activations 3D [N, 31, 64]; got {acts.shape}"
        )
    cell_types = [str(s) for s in d["cell_types"]]
    per_ct_means = acts.mean(axis=0)  # [31, 64]
    return per_ct_means, cell_types


def select_per_ct_top_features(
    sae: SAEModel,
    feature_reports: list[dict],
    per_ct_means: np.ndarray,
    cell_types_axis: list[str],
) -> list[dict]:
    """For each of 31 CTs, pick the top SAE feature.

    Selection criterion (mirrors the docstring spec):
      * Among the relaxed-pool features (323 in canonical SAE), find features
        whose ``top_cell_types[0]["cell_type"] == ct_name``. Pick the one
        with maximum ``ct_dominance``. → ``feature_source = "relaxed_pool"``.
      * If the relaxed pool yields zero matches for this CT, fall back to
        ALL non-dead features in the 2048-feature dictionary and pick the
        feature with maximum projection ``μ_CT @ W_dec[:, j]``. →
        ``feature_source = "fallback_full_2048"``.

    Returns
    -------
    list[dict] with one entry per CT (in ``CELL_TYPE_ORDER`` order):
        ``{"ct_index": int, "ct_name": str, "feature_idx": int,
           "feature_source": "relaxed_pool" | "fallback_full_2048",
           "ct_dominance": float, "selection_metric": float,
           "selection_metric_name": str}``
    """
    relaxed = _filter_relaxed(feature_reports)
    # Pre-bucket relaxed features by their top-1 CT for O(1) lookup per CT.
    relaxed_by_top_ct: dict[str, list[dict]] = {}
    for r in relaxed:
        tcts = r.get("top_cell_types") or []
        if not tcts:
            continue
        top_ct = tcts[0].get("cell_type")
        if top_ct is None:
            continue
        relaxed_by_top_ct.setdefault(top_ct, []).append(r)

    is_dead = sae.activation_stats.get("is_dead")
    if is_dead is None:
        non_dead_mask = np.ones(sae.W_enc.shape[0], dtype=bool)
    else:
        non_dead_mask = ~np.asarray(is_dead, dtype=bool)
    non_dead_idxs = np.where(non_dead_mask)[0]

    chosen: list[dict] = []
    for ct_idx, ct_name in enumerate(CELL_TYPE_ORDER):
        candidates = relaxed_by_top_ct.get(ct_name, [])
        if candidates:
            # Pick max ct_dominance within the relaxed-pool candidates.
            best = max(candidates, key=lambda r: float(r.get("ct_dominance", 0.0)))
            chosen.append({
                "ct_index": int(ct_idx),
                "ct_name": ct_name,
                "feature_idx": int(best["feature_idx"]),
                "feature_source": "relaxed_pool",
                "ct_dominance": float(best.get("ct_dominance", 0.0)),
                "selection_metric": float(best.get("ct_dominance", 0.0)),
                "selection_metric_name": "ct_dominance",
            })
        else:
            # Fallback: project per-CT mean activation vector onto every
            # non-dead decoder column and pick max.
            if ct_name not in cell_types_axis:
                raise ValueError(
                    f"CT {ct_name!r} not found in activation cell_types axis: "
                    f"{cell_types_axis!r}"
                )
            ct_axis_idx = cell_types_axis.index(ct_name)
            mu_ct = per_ct_means[ct_axis_idx]  # [n=64]
            # W_dec is [n, m]; project mu_ct onto every column.
            projections = mu_ct @ sae.W_dec  # [m]
            # Restrict to non-dead features.
            projections_nd = projections[non_dead_idxs]
            # Pick max squared projection (matches `interpret_features`'s
            # `np.argsort(-sq)` ranking; squared so sign-flipped decoder
            # directions are not penalised).
            best_local = int(np.argmax(projections_nd ** 2))
            best_global = int(non_dead_idxs[best_local])
            # Compute ct_dominance for the chosen feature: fraction of squared
            # projection mass on the chosen CT vs all CTs (matches the formula
            # at sparse_autoencoder.py:1258).
            decoder_col = sae.W_dec[:, best_global]
            proj_all_cts = per_ct_means @ decoder_col  # [C]
            sq = proj_all_cts ** 2
            total = float(sq.sum())
            ct_axis_for_dom = cell_types_axis.index(ct_name)
            ct_dominance = float(sq[ct_axis_for_dom] / total) if total > 0 else 0.0
            chosen.append({
                "ct_index": int(ct_idx),
                "ct_name": ct_name,
                "feature_idx": int(best_global),
                "feature_source": "fallback_full_2048",
                "ct_dominance": ct_dominance,
                "selection_metric": float(projections[best_global] ** 2),
                "selection_metric_name": "squared_projection",
            })
    return chosen


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
        help="Trained-SAE config directory (containing sae_model.npz, feature_report.json).",
    )
    p.add_argument(
        "--fused-activations-npz",
        default=DEFAULT_FUSED_ACTIVATIONS,
        help="Pooled-fold fused activations for percentile cutoff calculation.",
    )
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--n-random", type=int, default=10)
    p.add_argument("--random-seed", type=int, default=42)
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--out-dir",
        default="outputs/canonical/interpretability",
        help="Root for sae_causal_patching_31ct.json.",
    )
    p.add_argument(
        "--metadata-path", type=Path, default=None,
        help="Override config's data.metadata_path (the directory containing "
             "metadata.csv). Useful when the config ships placeholder paths.",
    )
    p.add_argument(
        "--precomputed-dir", type=Path, default=None,
        help="Override config's data.precomputed_dir (PrecomputedDataset cache "
             "of R*.pt files). Useful when the config ships placeholder paths.",
    )
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    # Determinism: same seed pattern as the original script.
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.random_seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    sae_dir = Path(args.sae_dir)
    sae = load_sae_from_dir(sae_dir)
    cfg_dict_for_log = sae.config
    logger.info(
        "Loaded SAE: arch=%s expansion=%s k=%s seed=%s from %s",
        cfg_dict_for_log.architecture, cfg_dict_for_log.expansion,
        cfg_dict_for_log.k, cfg_dict_for_log.seed, sae_dir,
    )

    if device.type == "cuda":
        logger.info(
            "Using device=%s (visible CUDA devices: %s)",
            device, torch.cuda.device_count(),
        )

    # ─── Per-CT top-feature selection ────────────────────────────────────────
    feature_reports = _load_feature_report(sae_dir)
    if len(feature_reports) != sae.W_enc.shape[0]:
        logger.warning(
            "Feature report length (%d) != SAE m (%d); proceeding with report",
            len(feature_reports), sae.W_enc.shape[0],
        )

    fused_activations_npz = Path(args.fused_activations_npz)
    per_ct_means, cell_types_axis = _compute_per_ct_means(fused_activations_npz)
    if cell_types_axis != list(CELL_TYPE_ORDER):
        # Reorder per_ct_means to match canonical CELL_TYPE_ORDER if needed.
        # The activations npz is expected to use the canonical order, so we
        # warn rather than silently reorder; reorder explicitly for safety.
        logger.warning(
            "Activation cell_types axis differs from CELL_TYPE_ORDER; reordering"
        )
        order_map = [cell_types_axis.index(ct) for ct in CELL_TYPE_ORDER]
        per_ct_means = per_ct_means[order_map]
        cell_types_axis = list(CELL_TYPE_ORDER)

    chosen_per_ct = select_per_ct_top_features(
        sae, feature_reports, per_ct_means, cell_types_axis,
    )
    n_relaxed = sum(1 for r in chosen_per_ct if r["feature_source"] == "relaxed_pool")
    n_fallback = sum(1 for r in chosen_per_ct if r["feature_source"] == "fallback_full_2048")
    logger.info(
        "Selected per-CT top features: %d from relaxed pool, %d from fallback",
        n_relaxed, n_fallback,
    )

    # Compute cohort percentiles for each chosen feature (saturate p99).
    chosen_feature_idxs = [r["feature_idx"] for r in chosen_per_ct]
    per_feature_cohort_stats: dict[int, dict[str, float]] = {}
    for r in chosen_per_ct:
        feat_idx = int(r["feature_idx"])
        if feat_idx in per_feature_cohort_stats:
            continue
        stats = compute_feature_percentiles(
            fused_activations_npz, sae,
            feature_idx=feat_idx, pct_low=1.0, pct_high=99.0,
        )
        per_feature_cohort_stats[feat_idx] = stats

    # ─── Random feature controls (10 features, IDENTICAL to original experiment) ──
    # Use the EXACT same selection (target_feature_idx=572, seed=42) AND patch
    # value (Splatter feature 572's p99) as run_sae_causal_patching.py so the
    # random_aggregate in this run's JSON matches the original
    # sae_causal_patching.json random_aggregate exactly. This serves as a
    # cross-experiment numerical verification (mean -7.3e-5, std 1.19e-3 at
    # n_random=10, random_seed=42).
    rng = np.random.default_rng(args.random_seed)
    random_feature_idxs = select_random_controls(
        sae,
        target_feature_idx=SPLATTER_FEATURE_IDX_LEGACY,
        n_random=int(args.n_random),
        rng=rng,
    ).tolist()
    # Patch value for ALL random controls is Splatter feat 572's p99 (matches
    # the existing experiment's `random_mode_value = p99` at line 645).
    splatter_legacy_stats = compute_feature_percentiles(
        fused_activations_npz, sae,
        feature_idx=SPLATTER_FEATURE_IDX_LEGACY,
        pct_low=1.0, pct_high=99.0,
    )
    random_patch_value = float(splatter_legacy_stats["p99"])
    logger.info(
        "Selected %d random control features (parity with original): %s "
        "(patch_value = Splatter feat 572 p99 = %.6f)",
        args.n_random, random_feature_idxs, random_patch_value,
    )
    # Warn if any random feature happens to be in the 31 chosen set (so the
    # reader knows the random null is not strictly disjoint from the per-CT set).
    chosen_set = set(chosen_feature_idxs)
    overlap = [i for i in random_feature_idxs if int(i) in chosen_set]
    if overlap:
        logger.warning(
            "%d random control features overlap with the 31 chosen per-CT "
            "features: %s. Random null is intentionally kept identical to the "
            "original experiment for numerical parity.",
            len(overlap), overlap,
        )

    # ─── Per-fold patching loop ──────────────────────────────────────────────
    splatter_ct_idx = list(CELL_TYPE_ORDER).index(SPLATTER_CT_NAME)
    fold_results: list[dict] = []
    cfg_full = OmegaConf.merge(
        OmegaConf.load(_WORKTREE_ROOT / "configs" / "default.yaml"),
        OmegaConf.load(_WORKTREE_ROOT / args.config),
    )
    OmegaConf.set_struct(cfg_full, False)
    cfg_full.model.head.type = "deterministic"
    if args.metadata_path is not None:
        cfg_full.data.metadata_path = str(args.metadata_path)
    if args.precomputed_dir is not None:
        cfg_full.data.precomputed_dir = str(args.precomputed_dir)

    fold_ckpt_paths: list[str] = []

    t_start = time.time()
    for fold in range(args.n_folds):
        fold_cfg = OmegaConf.create(OmegaConf.to_container(cfg_full, resolve=True))
        OmegaConf.set_struct(fold_cfg, False)
        fold_cfg.data.fold = int(fold)

        fold_dir = Path(args.canonical_dir) / f"fold{fold}"
        ckpt_path = pick_max_r2_ckpt(fold_dir / "checkpoints")
        fold_ckpt_paths.append(str(ckpt_path))
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
        tabpfn_map = _load_tabpfn_outer_map(Path(args.tabpfn_dir), int(fold))

        # Precompute fused + path_emb once per batch (single encoder forward
        # per batch shared across all 1 + 1 + 31 + 10 = 43 patch modes below).
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
            "fold %d: sae_baseline R²=%+.4f Δ_round_trip=%+.4f",
            fold, sae_baseline_r2, sae_baseline_r2 - enc_baseline_r2,
        )

        # 2) Per-CT top-feature saturate patches.
        per_ct_outputs: dict[int, dict] = {}
        for entry in chosen_per_ct:
            feat_idx = int(entry["feature_idx"])
            patch_value = float(per_feature_cohort_stats[feat_idx]["p99"])
            res = _gather_val_outputs(
                model, cached_batches, sae,
                mode="patch", feature_idx=feat_idx, patch_value=patch_value,
                tabpfn_map=tabpfn_map,
            )
            r2 = float(r2_score(res["truth"], res["composite"]))
            d_yhat = res["composite"] - sae_baseline_out["composite"]
            per_ct_outputs[entry["ct_index"]] = {
                "ct_name": entry["ct_name"],
                "feature_idx": feat_idx,
                "r2": r2,
                "delta_r2_vs_sae_baseline": r2 - sae_baseline_r2,
                "delta_r2_vs_encoder_baseline": r2 - enc_baseline_r2,
                "delta_yhat_mean": float(d_yhat.mean()),
                "delta_yhat_std": float(d_yhat.std(ddof=1)) if len(d_yhat) > 1 else 0.0,
            }
            logger.info(
                "fold %d CT=%-30s feat=%4d ΔR²=%+.4f",
                fold, entry["ct_name"][:30], feat_idx,
                r2 - sae_baseline_r2,
            )

        # 3) Random feature controls (saturate at the SHARED Splatter feat 572
        # p99 — identical to the original experiment so the random null
        # numerically reproduces the original sae_causal_patching.json values).
        random_outputs: dict[int, dict] = {}
        for ridx in random_feature_idxs:
            res = _gather_val_outputs(
                model, cached_batches, sae,
                mode="patch", feature_idx=int(ridx), patch_value=random_patch_value,
                tabpfn_map=tabpfn_map,
            )
            r2 = float(r2_score(res["truth"], res["composite"]))
            d_yhat_random = res["composite"] - sae_baseline_out["composite"]
            random_outputs[int(ridx)] = {
                "feature_idx": int(ridx),
                "r2": r2,
                "delta_r2_vs_sae_baseline": r2 - sae_baseline_r2,
                "delta_r2_vs_encoder_baseline": r2 - enc_baseline_r2,
                "delta_yhat_mean": float(d_yhat_random.mean()),
                "delta_yhat_std": float(d_yhat_random.std(ddof=1))
                    if len(d_yhat_random) > 1 else 0.0,
            }

        fold_results.append({
            "fold": int(fold),
            "n_val": int(len(enc_baseline_out["truth"])),
            "encoder_baseline_r2": enc_baseline_r2,
            "sae_baseline_r2": sae_baseline_r2,
            "per_ct": per_ct_outputs,
            "random_feature_controls": random_outputs,
        })

        del cached_batches, val_batches, model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    elapsed_min = (time.time() - t_start) / 60.0

    # ─── Aggregate across folds ──────────────────────────────────────────────
    encoder_per_fold = [fr["encoder_baseline_r2"] for fr in fold_results]
    sae_per_fold = [fr["sae_baseline_r2"] for fr in fold_results]

    per_ct_aggregate: list[dict] = []
    for entry in chosen_per_ct:
        ct_idx = int(entry["ct_index"])
        feat_idx = int(entry["feature_idx"])
        per_fold_records = []
        for fr in fold_results:
            d = fr["per_ct"][ct_idx]
            per_fold_records.append({
                "fold": int(fr["fold"]),
                "r2": d["r2"],
                "delta_r2": d["delta_r2_vs_sae_baseline"],
                "delta_r2_vs_encoder_baseline": d["delta_r2_vs_encoder_baseline"],
                "delta_yhat_mean": d["delta_yhat_mean"],
                "delta_yhat_std": d["delta_yhat_std"],
            })
        delta_r2_per_fold = [rec["delta_r2"] for rec in per_fold_records]
        mean_d = float(np.mean(delta_r2_per_fold))
        std_d = float(np.std(delta_r2_per_fold, ddof=1)) \
            if len(delta_r2_per_fold) > 1 else 0.0
        per_ct_aggregate.append({
            "ct_index": ct_idx,
            "ct_name": entry["ct_name"],
            "feature_idx": feat_idx,
            "feature_source": entry["feature_source"],
            "ct_dominance": entry["ct_dominance"],
            "selection_metric": entry["selection_metric"],
            "selection_metric_name": entry["selection_metric_name"],
            "feature_cohort_stats": per_feature_cohort_stats[feat_idx],
            "per_fold": per_fold_records,
            "saturate_delta_r2_mean": mean_d,
            "saturate_delta_r2_std": std_d,
        })

    # Random aggregate (saturate ΔR² flattened across folds × features).
    random_aggregate: dict[str, list[float]] = {}
    for ridx in random_feature_idxs:
        delta_r2_per_fold = [
            fr["random_feature_controls"][int(ridx)]["delta_r2_vs_sae_baseline"]
            for fr in fold_results
        ]
        random_aggregate[str(int(ridx))] = delta_r2_per_fold

    flat_random_delta_r2 = [v for lst in random_aggregate.values() for v in lst]
    random_saturate_mean = float(np.mean(flat_random_delta_r2)) \
        if flat_random_delta_r2 else float("nan")
    random_saturate_std = float(np.std(flat_random_delta_r2, ddof=1)) \
        if len(flat_random_delta_r2) > 1 else 0.0

    # Across-CT statistics on the saturate ΔR² means.
    ct_means = np.array([d["saturate_delta_r2_mean"] for d in per_ct_aggregate])
    abs_ct_means = np.abs(ct_means)
    if len(ct_means) > 0:
        max_idx = int(np.argmax(abs_ct_means))
        ct_with_max = per_ct_aggregate[max_idx]["ct_name"]
        max_abs = float(abs_ct_means[max_idx])
    else:
        ct_with_max = ""
        max_abs = float("nan")

    summary = {
        "experiment": "sae_causal_patching_31ct",
        "method": "saturate-mode patch on top SAE feature per CT",
        "sae_config": {
            "dir": str(sae_dir),
            "architecture": cfg_dict_for_log.architecture,
            "expansion": int(cfg_dict_for_log.expansion),
            "k": int(cfg_dict_for_log.k) if cfg_dict_for_log.k is not None else None,
            "seed": int(cfg_dict_for_log.seed),
            "layer": "fused",
        },
        "feature_selection": {
            "criterion_relaxed_pool": (
                "non-dead AND mw_p_cognition < 0.05 AND fraction_active in "
                f"[{DEAD_FRACTION_THRESHOLD}, 0.5]; pick max ct_dominance among "
                "features with top_cell_types[0] == this_CT"
            ),
            "criterion_fallback": (
                "non-dead in full 2048-feature dictionary; pick max squared "
                "projection (mu_CT @ W_dec[:, j])^2"
            ),
            "n_relaxed_pool_total": len(_filter_relaxed(feature_reports)),
            "n_chosen_relaxed_pool": int(sum(
                1 for d in per_ct_aggregate
                if d["feature_source"] == "relaxed_pool"
            )),
            "n_chosen_fallback": int(sum(
                1 for d in per_ct_aggregate
                if d["feature_source"] == "fallback_full_2048"
            )),
        },
        "n_folds": int(args.n_folds),
        "n_cell_types": int(N_CELL_TYPES),
        "n_random_controls": int(args.n_random),
        "random_seed": int(args.random_seed),
        "encoder_baseline_per_fold_r2": encoder_per_fold,
        "encoder_baseline_mean_r2": float(np.mean(encoder_per_fold)),
        "sae_baseline_per_fold_r2": sae_per_fold,
        "sae_baseline_mean_r2": float(np.mean(sae_per_fold)),
        "per_ct": per_ct_aggregate,
        "random_feature_aggregate": {
            "patch_mode": "saturate",
            "patch_value_shared": random_patch_value,
            "patch_value_source": (
                f"feature {SPLATTER_FEATURE_IDX_LEGACY} (Splatter relaxed-pool "
                "feature) cohort p99; matches original sae_causal_patching.json"
            ),
            "feature_idxs": random_feature_idxs,
            "splatter_legacy_cohort_stats": splatter_legacy_stats,
            "delta_r2_per_fold": random_aggregate,
            "n_overlap_with_chosen_per_ct": int(len(overlap)),
            "overlap_feature_idxs": list(overlap),
        },
        "summary_statistics": {
            "across_ct_mean_delta_r2": float(np.mean(ct_means))
                if len(ct_means) > 0 else float("nan"),
            "across_ct_std_delta_r2": float(np.std(ct_means, ddof=1))
                if len(ct_means) > 1 else 0.0,
            "across_ct_mean_abs_delta_r2": float(np.mean(abs_ct_means))
                if len(abs_ct_means) > 0 else float("nan"),
            "max_abs_ct_delta_r2": max_abs,
            "ct_with_max_abs_delta_r2": ct_with_max,
            "random_saturate_delta_r2_mean": random_saturate_mean,
            "random_saturate_delta_r2_std": random_saturate_std,
            "n_random_pooled": int(len(flat_random_delta_r2)),
        },
        "provenance": {
            "git_commit": git_sha(_WORKTREE_ROOT),
            "n_random_controls": int(args.n_random),
            "fold_checkpoints": fold_ckpt_paths,
            "sae_checkpoint": str(sae_dir / "sae_model.npz"),
            "device": str(device),
            "wall_time_min": round(elapsed_min, 2),
            "fused_activations_npz": str(fused_activations_npz),
        },
    }

    # JSON output.
    json_path = out_dir / "sae_causal_patching_31ct.json"
    json_path.write_text(json.dumps(summary, indent=2, default=str))
    logger.info("wrote %s", json_path)

    # Stdout verification dump.
    print()
    print("=" * 88)
    print("  31-CT SAE Causal Patching Summary")
    print("=" * 88)
    print(
        f"SAE config: {cfg_dict_for_log.architecture}/fused/"
        f"exp{cfg_dict_for_log.expansion}_k{cfg_dict_for_log.k}_seed{cfg_dict_for_log.seed}"
    )
    print(
        f"Feature-selection: relaxed-pool={summary['feature_selection']['n_chosen_relaxed_pool']}/"
        f"31 (of {summary['feature_selection']['n_relaxed_pool_total']} pool); "
        f"fallback={summary['feature_selection']['n_chosen_fallback']}/31"
    )
    print(f"Encoder baseline mean R² = {summary['encoder_baseline_mean_r2']:+.4f}")
    print(f"SAE baseline mean R²     = {summary['sae_baseline_mean_r2']:+.4f}")
    print()
    print(
        f"{'CT':<35s}  {'feat':>5s}  {'src':<10s}  "
        f"{'ct_dom':>7s}  {'ΔR² mean':>10s}  {'ΔR² std':>9s}"
    )
    print("-" * 88)
    for d in per_ct_aggregate:
        src_short = "relaxed" if d["feature_source"] == "relaxed_pool" else "fallback"
        print(
            f"{d['ct_name'][:35]:<35s}  {d['feature_idx']:>5d}  "
            f"{src_short:<10s}  {d['ct_dominance']:>7.4f}  "
            f"{d['saturate_delta_r2_mean']:>+10.6f}  "
            f"{d['saturate_delta_r2_std']:>9.6f}"
        )
    print("-" * 88)
    print(
        f"Across-CT mean ΔR²       = {summary['summary_statistics']['across_ct_mean_delta_r2']:+.6f} "
        f"(std {summary['summary_statistics']['across_ct_std_delta_r2']:.6f})"
    )
    print(
        f"Across-CT mean |ΔR²|     = {summary['summary_statistics']['across_ct_mean_abs_delta_r2']:.6f}"
    )
    print(
        f"Max |ΔR²| across CTs     = {max_abs:.6f}  ({ct_with_max})"
    )
    print(
        f"Random saturate ΔR² mean = "
        f"{random_saturate_mean:+.6f} ± {random_saturate_std:.6f} "
        f"(n={len(flat_random_delta_r2)})"
    )
    print(f"Wall time                = {elapsed_min:.2f} min")
    print(f"Device used              = {device}")
    print("=" * 88)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
