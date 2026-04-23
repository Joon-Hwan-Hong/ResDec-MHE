"""5-fold CCC interpretability sweep for ResDec-H3 canonical model.

For each fold:

1. Loads the max-R² ``best-*.ckpt`` from ``<pred-root>/fold{f}/checkpoints/``.
2. Runs one val-set forward pass with ``return_hgt_attention=True`` to extract
   HGT edge attention, aggregated at two levels:

   - per-edge-type (5-way): Secreted / ECM / Cell-Cell / Non-protein / Novel.
   - per-(source_ct, target_ct, edge_type): 31 × 31 × 5 triples, averaged over
     all HGT layers + attention heads + batches in the val set.

3. Runs per-edge-type ablation — 5 extra val passes, each with one edge type's
   rows dropped from every batch. Reports R² delta (baseline − ablated).

4. Aggregates (source_ct, target_ct, edge_type) attention across folds, then
   correlates the per-(source, target) summary against LIANA's CellChatDB
   reference (mean ``magnitude_rank`` over the same subject set).

Outputs (default ``outputs/redesign/interpretability/ccc/``):

- ``ccc_importance.json``       — baseline / ablated R² per fold + aggregated
                                  per-edge-type attention + LIANA correlation.
- ``ccc_ablation_table.csv``    — per-fold × edge-type R² deltas (long format).
- ``ccc_edge_attention.csv``    — per-fold × (source, target, edge_type) mean
                                  attention (long format, aggregated across subjects).
- ``ccc_celltype_pair_importance.csv``
                                — cross-fold aggregated (source_ct, target_ct) importance
                                  + LIANA score (joined), used for correlation.

Usage
-----
::

    PYTHONPATH=<worktree-root> \\
    CUDA_VISIBLE_DEVICES=0 \\
    uv run python scripts/redesign/interpretability/ccc_composite_attribution.py \\
        --pred-root outputs/redesign/p5_canonical_seed42 \\
        --tabpfn-dir data/redesign \\
        --liana-dir data/liana_cache/rosmap \\
        --out-dir outputs/redesign/interpretability/ccc
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

_WORKTREE_ROOT = Path(__file__).resolve().parents[3]
if not (_WORKTREE_ROOT / "src").is_dir():
    raise RuntimeError(
        f"sys.path bootstrap failed: {_WORKTREE_ROOT}/src not found; "
        "set PYTHONPATH=<worktree-root>."
    )
sys.path.insert(0, str(_WORKTREE_ROOT))

from src.analysis.resdec_ccc_importance import (  # noqa: E402
    aggregate_attention_by_celltype_pair,
    aggregate_attention_by_edge_type,
    liana_correlation,
    load_liana_reference,
    per_edge_type_ablation,
)
from src.data.constants import (  # noqa: E402
    ALL_EDGE_TYPES,
    CELL_TYPE_ORDER,
    EDGE_TYPE_DISPLAY_NAMES,
    N_EDGE_TYPES,
)
from src.data.datamodule import CognitiveResilienceDataModule  # noqa: E402
from src.data.splits import load_splits  # noqa: E402
from src.training.resdec_lightning_module import ResDecLightningModule  # noqa: E402

logger = logging.getLogger(__name__)
_BEST_CKPT_RE = re.compile(r"^best-(\d+)-(\d+\.\d+)\.ckpt$")


def _pick_max_r2_ckpt(ckpt_dir: Path) -> Path:
    best: tuple[Path, float] | None = None
    for p in ckpt_dir.glob("best-*.ckpt"):
        m = _BEST_CKPT_RE.match(p.name)
        if not m:
            continue
        r2 = float(m.group(2))
        if best is None or r2 > best[1]:
            best = (p, r2)
    if best is None:
        raise FileNotFoundError(f"No best-*.ckpt files in {ckpt_dir}")
    return best[0]


def _move_batch(b: dict, device: torch.device) -> dict:
    out: dict = {}
    for k, v in b.items():
        if torch.is_tensor(v):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def analyze_one_fold(
    args: argparse.Namespace,
    fold: int,
    device: torch.device,
) -> dict:
    """Run the full CCC analysis on one fold: load ckpt → extract → ablate."""
    cfg = OmegaConf.merge(
        OmegaConf.load("configs/default.yaml"),
        OmegaConf.load(args.config),
    )
    OmegaConf.set_struct(cfg, False)
    cfg.model.head.type = "deterministic"
    cfg.data.fold = int(fold)

    fold_dir = Path(args.pred_root) / f"fold{fold}"
    ckpt_path = _pick_max_r2_ckpt(fold_dir / "checkpoints")
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

    lit_module = ResDecLightningModule.load_from_checkpoint(
        str(ckpt_path), config=cfg, map_location="cpu",
    ).to(device).eval()
    # Cast to fp32: attention accumulation in bf16 is too lossy for per-edge-type
    # means (attention weights ~1/n_edges; bf16 mantissa = 7 bits).
    lit_module = lit_module.float()

    val_loader = dm.val_dataloader()
    n_edge_types = N_EDGE_TYPES  # 5
    n_nodes_per_graph = len(CELL_TYPE_ORDER)  # 31

    # ---------------------------------------------------------------- #
    # 1. Walk val loader once with return_hgt_attention=True           #
    #    + aggregate per-edge-type attention + per-pair attention.     #
    #    We also capture per-subject "importance" as mean attention    #
    #    over (source_ct, target_ct) triples for LIANA correlation.    #
    # ---------------------------------------------------------------- #
    per_type_sums = np.zeros(n_edge_types, dtype=np.float64)
    per_type_counts = np.zeros(n_edge_types, dtype=np.int64)

    pair_frames: list[pd.DataFrame] = []
    val_subject_ids: list[str] = []

    logger.info("fold %d: extracting HGT edge attention from val set", fold)
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"fold {fold} attention", unit="batch"):
            batch = _move_batch(batch, device)
            enc_kwargs = {k: batch.get(k) for k in (
                "region_pseudobulk", "region_mask", "pseudobulk",
                "ccc_edge_index", "ccc_edge_type", "ccc_edge_attr",
                "cell_type_mask", "pathology", "cognition",
                "cell_data", "cell_offsets",
            ) if k in batch}
            enc_kwargs["return_hgt_attention"] = True
            enc_out = lit_module.encoder(**enc_kwargs)

            attn_list = enc_out.get("hgt_attention")
            if not attn_list:
                raise RuntimeError("Encoder didn't return hgt_attention — check model contract")

            # Stack → [n_layers, E, H] → average over layers + heads → [E]
            attn_stack = torch.stack([a.detach().float() for a in attn_list], dim=0)
            attn_mean = attn_stack.mean(dim=(0, -1))  # [E]

            et = batch["ccc_edge_type"]
            if et.numel() > 0:
                # Per-edge-type sums (for cross-batch mean).
                et_np = et.detach().cpu().numpy()
                attn_np = attn_mean.detach().cpu().numpy()
                for k in range(n_edge_types):
                    mask = et_np == k
                    per_type_counts[k] += int(mask.sum())
                    per_type_sums[k] += float(attn_np[mask].sum())

                # Per (source_ct, target_ct, edge_type) aggregation.
                pair_df = aggregate_attention_by_celltype_pair(
                    attention=attn_mean.unsqueeze(-1),  # [E, 1 "head"]
                    edge_index=batch["ccc_edge_index"],
                    edge_type=batch["ccc_edge_type"],
                    n_nodes_per_graph=n_nodes_per_graph,
                )
                pair_df["fold"] = fold
                # Weight the per-batch means by n_edges so cross-batch averaging
                # is equivalent to averaging over the full flat edge set.
                pair_df["weighted_sum"] = pair_df["mean_attention"] * pair_df["n_edges"]
                pair_frames.append(pair_df)

            val_subject_ids.extend(list(batch["subject_ids"]))

    # Cross-batch per-edge-type mean
    with np.errstate(invalid="ignore", divide="ignore"):
        per_type_mean = np.where(
            per_type_counts > 0,
            per_type_sums / np.maximum(per_type_counts, 1),
            np.nan,
        )

    # Cross-batch per-pair aggregation: re-group on the concatenated frame.
    if pair_frames:
        all_pairs = pd.concat(pair_frames, ignore_index=True)
        pair_agg = (
            all_pairs.groupby(["source_ct_idx", "target_ct_idx", "edge_type"], as_index=False)
            .agg(
                weighted_sum=("weighted_sum", "sum"),
                n_edges=("n_edges", "sum"),
            )
        )
        pair_agg["mean_attention"] = pair_agg["weighted_sum"] / pair_agg["n_edges"]
        pair_agg["fold"] = fold
        pair_agg["source_ct"] = pair_agg["source_ct_idx"].map(
            lambda i: CELL_TYPE_ORDER[int(i)] if 0 <= int(i) < len(CELL_TYPE_ORDER) else f"ct_{i}"
        )
        pair_agg["target_ct"] = pair_agg["target_ct_idx"].map(
            lambda i: CELL_TYPE_ORDER[int(i)] if 0 <= int(i) < len(CELL_TYPE_ORDER) else f"ct_{i}"
        )
        pair_agg["edge_type_name"] = pair_agg["edge_type"].map(
            lambda i: ALL_EDGE_TYPES[int(i)] if 0 <= int(i) < len(ALL_EDGE_TYPES) else f"et_{i}"
        )
    else:
        pair_agg = pd.DataFrame(
            columns=[
                "source_ct_idx", "target_ct_idx", "edge_type",
                "weighted_sum", "n_edges", "mean_attention", "fold",
                "source_ct", "target_ct", "edge_type_name",
            ]
        )

    logger.info(
        "fold %d: per-edge-type attention mean = %s (counts=%s)",
        fold,
        np.round(per_type_mean, 4).tolist(),
        per_type_counts.tolist(),
    )

    # ---------------------------------------------------------------- #
    # 2. Per-edge-type ablation sweep (1 + n_edge_types passes).       #
    # ---------------------------------------------------------------- #
    logger.info("fold %d: running per-edge-type ablation", fold)
    ablation = per_edge_type_ablation(
        lit_module=lit_module,
        val_dataloader=val_loader,
        n_edge_types=n_edge_types,
        device=device,
        edge_type_names=list(ALL_EDGE_TYPES),
    )

    # ---------------------------------------------------------------- #
    # 3. Release GPU memory before the next fold.                      #
    # ---------------------------------------------------------------- #
    del lit_module, dm, val_loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "fold": fold,
        "ckpt_path": str(ckpt_path),
        "val_subject_ids": val_subject_ids,
        "per_edge_type_attention_mean": per_type_mean.tolist(),
        "per_edge_type_edge_counts": per_type_counts.tolist(),
        "pair_agg": pair_agg,
        "ablation": ablation,
    }


def main(args: argparse.Namespace) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)
    logger.info("Edge types: %s", ALL_EDGE_TYPES)

    # ──────────────────────────────────────────────────────────── #
    # Per-fold sweep                                               #
    # ──────────────────────────────────────────────────────────── #
    fold_results: list[dict] = []
    for f in range(int(args.n_folds)):
        fr = analyze_one_fold(args, f, device)
        fold_results.append(fr)
        logger.info(
            "fold %d done: baseline R²=%.4f",
            f, fr["ablation"]["baseline_r2"],
        )

    # ──────────────────────────────────────────────────────────── #
    # Aggregate ablation table (per-fold × edge-type)               #
    # ──────────────────────────────────────────────────────────── #
    abl_rows: list[dict] = []
    for fr in fold_results:
        for row in fr["ablation"]["per_edge_type"]:
            abl_rows.append({
                "fold": fr["fold"],
                "edge_type_idx": row["edge_type_idx"],
                "edge_type_name": row["edge_type_name"],
                "edge_type_display": EDGE_TYPE_DISPLAY_NAMES.get(
                    row["edge_type_name"], row["edge_type_name"]
                ),
                "baseline_r2": fr["ablation"]["baseline_r2"],
                "ablated_r2": row["ablated_r2"],
                "r2_delta": row["r2_delta"],
                "n_edges_ablated": row["n_edges_ablated"],
            })
    abl_df = pd.DataFrame(abl_rows)
    abl_csv = out_dir / "ccc_ablation_table.csv"
    abl_df.to_csv(abl_csv, index=False)
    logger.info("Wrote %s (%d rows)", abl_csv, len(abl_df))

    # Per-edge-type aggregates across folds (mean R² delta).
    abl_summary = (
        abl_df.groupby(["edge_type_idx", "edge_type_name", "edge_type_display"], as_index=False)
        .agg(
            mean_baseline_r2=("baseline_r2", "mean"),
            mean_ablated_r2=("ablated_r2", "mean"),
            mean_r2_delta=("r2_delta", "mean"),
            std_r2_delta=("r2_delta", "std"),
            total_edges_ablated=("n_edges_ablated", "sum"),
        )
        .sort_values("mean_r2_delta", ascending=False)
    )

    # ──────────────────────────────────────────────────────────── #
    # Aggregate edge-attention per fold × (source, target, type)    #
    # ──────────────────────────────────────────────────────────── #
    all_pairs = pd.concat(
        [fr["pair_agg"] for fr in fold_results if not fr["pair_agg"].empty],
        ignore_index=True,
    )
    attn_csv = out_dir / "ccc_edge_attention.csv"
    all_pairs.to_csv(attn_csv, index=False)
    logger.info("Wrote %s (%d rows)", attn_csv, len(all_pairs))

    # ──────────────────────────────────────────────────────────── #
    # Cross-fold aggregation at (source_ct, target_ct) level        #
    # (LIANA's reference doesn't split by edge_type; we sum attention
    #  over edge types to align with LIANA's pair-level score).     #
    # ──────────────────────────────────────────────────────────── #
    if not all_pairs.empty:
        pair_ct = (
            all_pairs.groupby(["source_ct", "target_ct"], as_index=False)
            .agg(
                importance=("mean_attention", "mean"),
                total_edges=("n_edges", "sum"),
                n_folds_seen=("fold", "nunique"),
            )
        )
    else:
        pair_ct = pd.DataFrame(columns=["source_ct", "target_ct", "importance", "total_edges", "n_folds_seen"])

    # ──────────────────────────────────────────────────────────── #
    # LIANA reference + correlation                                 #
    # ──────────────────────────────────────────────────────────── #
    logger.info("Loading LIANA reference from %s", args.liana_dir)
    # Union all val subjects across folds == every subject once (5-fold CV).
    all_val_subjects = sorted(set().union(*[fr["val_subject_ids"] for fr in fold_results]))
    liana_full = load_liana_reference(
        liana_dir=Path(args.liana_dir),
        subject_ids=all_val_subjects,
        score_col=args.liana_score_col,
    )
    logger.info(
        "LIANA: loaded %d rows from %d subjects (score_col=%s)",
        len(liana_full), liana_full["subject_id"].nunique(), args.liana_score_col,
    )

    # Correlation: our per-(source_ct, target_ct) importance ↔ LIANA score.
    liana_corr = liana_correlation(
        our_ranking=pair_ct.rename(columns={"importance": "importance"})[
            ["source_ct", "target_ct", "importance"]
        ],
        liana_df=liana_full,
        score_col=args.liana_score_col,
        higher_is_better=False,  # magnitude_rank & specificity_rank: lower = better
    )
    logger.info(
        "LIANA correlation: Pearson=%.4f, Spearman=%.4f, n_pairs=%d, n_missing=%d",
        liana_corr["pearson_r"], liana_corr["spearman_rho"],
        liana_corr["n_pairs"], liana_corr["n_missing"],
    )

    # Save joined per-pair table (our importance + LIANA score) for plotting.
    if not pair_ct.empty:
        liana_pair = (
            liana_full.groupby(["source", "target"], as_index=False)[args.liana_score_col]
            .mean()
            .rename(columns={"source": "source_ct", "target": "target_ct"})
        )
        joined = pair_ct.merge(liana_pair, on=["source_ct", "target_ct"], how="left")
        joined = joined.rename(columns={args.liana_score_col: f"liana_{args.liana_score_col}"})
        joined_csv = out_dir / "ccc_celltype_pair_importance.csv"
        joined.to_csv(joined_csv, index=False)
        logger.info("Wrote %s (%d rows)", joined_csv, len(joined))

    # ──────────────────────────────────────────────────────────── #
    # Consolidated JSON report                                      #
    # ──────────────────────────────────────────────────────────── #
    report = {
        "provenance": {
            "pred_root": str(args.pred_root),
            "tabpfn_dir": str(args.tabpfn_dir),
            "liana_dir": str(args.liana_dir),
            "liana_score_col": args.liana_score_col,
            "n_folds": int(args.n_folds),
            "edge_types": list(ALL_EDGE_TYPES),
            "edge_type_display_names": {
                k: EDGE_TYPE_DISPLAY_NAMES.get(k, k) for k in ALL_EDGE_TYPES
            },
            "n_cell_types": len(CELL_TYPE_ORDER),
            "ckpt_paths": {
                str(fr["fold"]): fr["ckpt_path"] for fr in fold_results
            },
            "val_subjects_union_n": len(all_val_subjects),
        },
        "per_fold": [
            {
                "fold": fr["fold"],
                "baseline_r2": fr["ablation"]["baseline_r2"],
                "baseline_n": fr["ablation"]["baseline_n"],
                "per_edge_type_attention_mean": fr["per_edge_type_attention_mean"],
                "per_edge_type_edge_counts": fr["per_edge_type_edge_counts"],
                "ablation": fr["ablation"]["per_edge_type"],
            }
            for fr in fold_results
        ],
        "ablation_summary_across_folds": abl_summary.to_dict(orient="records"),
        "liana_correlation": liana_corr,
    }
    report_path = out_dir / "ccc_importance.json"
    report_path.write_text(json.dumps(report, indent=2, default=_jsonify))
    logger.info("Wrote %s", report_path)

    # Human-readable stdout summary.
    print()
    print("=== Per-edge-type ablation (averaged across folds) ===")
    for r in abl_summary.itertuples():
        print(
            f"  [{r.edge_type_idx}] {r.edge_type_display:<24s}"
            f"  mean ΔR²={r.mean_r2_delta:+.4f} ± {r.std_r2_delta:.4f}"
            f"  (n_edges_total={r.total_edges_ablated})"
        )
    print()
    print("=== LIANA correlation (our importance vs LIANA "
          f"{args.liana_score_col}) ===")
    print(f"  Pearson r    : {liana_corr['pearson_r']:.4f}")
    print(f"  Spearman ρ   : {liana_corr['spearman_rho']:.4f}")
    print(f"  n_pairs      : {liana_corr['n_pairs']}")
    print(f"  n_missing    : {liana_corr['n_missing']}")

    return 0


def _jsonify(o):
    """Fallback encoder for NumPy scalars and arrays."""
    if isinstance(o, (np.floating, np.integer)):
        return float(o) if isinstance(o, np.floating) else int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Cannot JSON-serialize {type(o)}: {o!r}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="5-fold CCC interpretability sweep")
    p.add_argument("--config", default="configs/redesign/p5_phase2_residual.yaml")
    p.add_argument("--pred-root", default="outputs/redesign/p5_canonical_seed42")
    p.add_argument("--tabpfn-dir", default="data/redesign")
    p.add_argument("--liana-dir", default="data/liana_cache/rosmap")
    p.add_argument("--splits-path", default="outputs/splits.json")
    p.add_argument("--out-dir", default="outputs/redesign/interpretability/ccc")
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument(
        "--liana-score-col",
        default="magnitude_rank",
        choices=["magnitude_rank", "specificity_rank", "lrscore", "lr_means"],
        help="LIANA column to correlate against. 'magnitude_rank' / "
             "'specificity_rank' are percentile ranks (lower = better); "
             "'lrscore' / 'lr_means' are magnitude-style (higher = better).",
    )
    sys.exit(main(p.parse_args()))
