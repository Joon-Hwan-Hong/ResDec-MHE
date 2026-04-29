"""Per-subject CCC HGT edge attention extraction (canonical ResDec-MHE).

Audit Finding 9 follow-up: the existing ``ccc_edge_attention.csv`` aggregates
attention across all subjects in each fold (population mean per (source-CT,
target-CT, edge-type)). This script disaggregates: for each fold, runs forward
on val subjects with ``return_hgt_attention=True`` and stores per-subject
HGT edge attention at the (source-CT, target-CT, edge-type) level.

For each subject we compute:
  - mean attention per (source-CT, target-CT, edge-type) on that subject's edges
  - top-5 edges (CT-pair, edge-type) by attention magnitude
  - max attention across all edges

The orchestrator then aggregates across the 5 folds (= union of all 516 val
subjects) and reports:
  - distribution of max-edge-attention across subjects (mean, p95, p99)
  - # subjects with at least one edge > 0.01 attention (5x the population mean)
  - top-3 most-frequent high-attention edges across subjects
  - per-subject top-5 edges (full table)

Outputs (under ``--out-dir``):
  - ``per_subject_ccc_attention_summary.json`` — aggregate summary
  - ``per_subject_ccc_attention.npz`` — per-subject (CT-pair, edge-type) attention
                                        tensor + subject IDs

Usage::

    PYTHONPATH=<worktree-root> \\
    CUDA_VISIBLE_DEVICES=0 \\
    uv run python scripts/resdec_mhe/interpretability/per_subject_ccc_attention.py \\
        --pred-root outputs/canonical/p5_canonical_seed42 \\
        --out-dir outputs/canonical/interpretability/ccc
"""
from __future__ import annotations

import argparse
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

from src.data.constants import (
    ALL_EDGE_TYPES,
    CELL_TYPE_ORDER,
    N_EDGE_TYPES,
)
from src.data.datamodule import CognitiveResilienceDataModule
from src.data.splits import load_splits
from src.training.resdec_lightning_module import (
    ENCODER_KWARG_KEYS,
    ResDecLightningModule,
)

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
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}


def _per_subject_attention_for_batch(
    encoder: torch.nn.Module,
    batch: dict,
    device: torch.device,
    n_edge_types: int,
    n_nodes_per_graph: int,
) -> tuple[np.ndarray, list[str]]:
    """Return [B, n_ct, n_ct, n_edge_types] mean attention per subject (NaN where absent)."""
    kwargs = {k: batch.get(k) for k in ENCODER_KWARG_KEYS if k in batch}
    kwargs["return_hgt_attention"] = True
    with torch.no_grad():
        out = encoder(**kwargs)

    attn_list = out.get("hgt_attention")
    if attn_list is None or len(attn_list) == 0:
        raise RuntimeError(
            "Encoder did not return 'hgt_attention'; verify the model propagates "
            "return_hgt_attention=True."
        )

    # Stack per-layer [E_total, H] → [n_layers, E_total, H], head- and layer-mean → [E_total]
    attn_stack = torch.stack(
        [a.detach().to(device=device, dtype=torch.float32) for a in attn_list], dim=0
    )
    attn_per_edge = attn_stack.mean(dim=-1).mean(dim=0).cpu().numpy()  # [E_total]

    edge_index = batch["ccc_edge_index"].detach().cpu().numpy()  # [2, E_total]
    edge_type = batch["ccc_edge_type"].detach().cpu().numpy()    # [E_total]

    # Subject id per edge: source-side floor-divide gives within-batch subject index.
    subj_of_edge = edge_index[0] // n_nodes_per_graph             # [E_total]
    src_ct = edge_index[0] % n_nodes_per_graph
    tgt_ct = edge_index[1] % n_nodes_per_graph

    sids = list(batch["subject_ids"])
    B = len(sids)

    # Accumulate sum + count per (subject, src_ct, tgt_ct, edge_type)
    attn_sum = np.zeros((B, n_nodes_per_graph, n_nodes_per_graph, n_edge_types), dtype=np.float64)
    counts = np.zeros((B, n_nodes_per_graph, n_nodes_per_graph, n_edge_types), dtype=np.int64)
    np.add.at(attn_sum, (subj_of_edge, src_ct, tgt_ct, edge_type), attn_per_edge)
    np.add.at(counts, (subj_of_edge, src_ct, tgt_ct, edge_type), 1)

    with np.errstate(invalid="ignore", divide="ignore"):
        mean_attn = np.where(counts > 0, attn_sum / np.maximum(counts, 1), np.nan)

    return mean_attn.astype(np.float32), sids


def extract_per_subject_attention(args: argparse.Namespace, device: torch.device) -> dict:
    """5-fold sweep returning per-subject mean attention + subject IDs."""
    n_edge_types = N_EDGE_TYPES
    n_nodes_per_graph = len(CELL_TYPE_ORDER)

    all_attn: list[np.ndarray] = []
    all_sids: list[str] = []
    all_folds: list[int] = []

    for fold in range(int(args.n_folds)):
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
        metadata = pd.read_csv(Path(cfg.data.metadata_path) / "metadata.csv")
        dm = CognitiveResilienceDataModule(
            config=cfg, metadata=metadata, splits=splits,
            fold_idx=fold,
            precomputed_dir=cfg.data.precomputed_dir,
            adata=None,
        )
        dm.setup(stage="fit")

        lit = ResDecLightningModule.load_from_checkpoint(
            str(ckpt_path), config=cfg, map_location="cpu",
        ).to(device).eval().float()

        for batch in tqdm(dm.val_dataloader(), desc=f"fold {fold}", unit="batch"):
            b = _move_batch(batch, device)
            mean_attn, sids = _per_subject_attention_for_batch(
                lit.encoder, b, device, n_edge_types, n_nodes_per_graph,
            )
            all_attn.append(mean_attn)
            all_sids.extend(sids)
            all_folds.extend([fold] * len(sids))

        del lit, dm
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    attn_arr = np.concatenate(all_attn, axis=0)  # [N_total, n_ct, n_ct, n_edge_types]
    return {
        "attention": attn_arr,
        "subject_ids": np.array(all_sids),
        "folds": np.array(all_folds, dtype=np.int32),
    }


def summarize(
    attn_arr: np.ndarray,  # [N, n_ct, n_ct, n_edge_types]
    subject_ids: np.ndarray,
    folds: np.ndarray,
    threshold: float,
    top_k: int,
) -> dict:
    """Build per-subject + cross-subject summary dicts."""
    n_ct = len(CELL_TYPE_ORDER)
    n_et = N_EDGE_TYPES
    N = attn_arr.shape[0]

    # Per-subject max edge attention (ignore NaN cells = absent edge types).
    with np.errstate(invalid="ignore"):
        per_subj_max = np.nanmax(attn_arr.reshape(N, -1), axis=1)

    # Per-subject top-k edges (by attention).
    per_subj_top: list[dict] = []
    high_attn_records: list[tuple[str, str, str, float, str]] = []
    for i in range(N):
        flat = attn_arr[i].reshape(-1)  # [n_ct * n_ct * n_et]
        finite_mask = np.isfinite(flat)
        if not finite_mask.any():
            per_subj_top.append({
                "subject_id": str(subject_ids[i]),
                "fold": int(folds[i]),
                "max_attention": float("nan"),
                "n_high_attention_edges": 0,
                "top_edges": [],
            })
            continue
        # Top-k indices (descending); set NaN to -inf to exclude.
        flat_for_sort = np.where(finite_mask, flat, -np.inf)
        top_idx = np.argsort(-flat_for_sort)[:top_k]
        top_edges = []
        for idx in top_idx:
            attn_val = float(flat[idx])
            if not np.isfinite(attn_val):
                continue
            src = idx // (n_ct * n_et)
            rem = idx - src * (n_ct * n_et)
            tgt = rem // n_et
            et = rem - tgt * n_et
            top_edges.append({
                "source_ct": CELL_TYPE_ORDER[int(src)],
                "target_ct": CELL_TYPE_ORDER[int(tgt)],
                "edge_type": ALL_EDGE_TYPES[int(et)],
                "attention": attn_val,
            })

        # Count edges over threshold for this subject.
        n_high = int(np.sum(np.where(finite_mask, flat > threshold, False)))

        # Record edges over threshold for cross-subject frequency tallying.
        if n_high > 0:
            over_idx = np.where(finite_mask & (flat > threshold))[0]
            for idx in over_idx:
                src = idx // (n_ct * n_et)
                rem = idx - src * (n_ct * n_et)
                tgt = rem // n_et
                et = rem - tgt * n_et
                high_attn_records.append((
                    CELL_TYPE_ORDER[int(src)],
                    CELL_TYPE_ORDER[int(tgt)],
                    ALL_EDGE_TYPES[int(et)],
                    float(flat[idx]),
                    str(subject_ids[i]),
                ))

        per_subj_top.append({
            "subject_id": str(subject_ids[i]),
            "fold": int(folds[i]),
            "max_attention": float(per_subj_max[i]),
            "n_high_attention_edges": n_high,
            "top_edges": top_edges,
        })

    # Distribution of max-edge-attention across subjects.
    finite_max = per_subj_max[np.isfinite(per_subj_max)]
    max_dist = {
        "n_subjects": int(N),
        "n_finite": int(finite_max.size),
        "mean": float(finite_max.mean()) if finite_max.size else float("nan"),
        "std": float(finite_max.std(ddof=1)) if finite_max.size > 1 else float("nan"),
        "min": float(finite_max.min()) if finite_max.size else float("nan"),
        "p25": float(np.percentile(finite_max, 25)) if finite_max.size else float("nan"),
        "median": float(np.median(finite_max)) if finite_max.size else float("nan"),
        "p75": float(np.percentile(finite_max, 75)) if finite_max.size else float("nan"),
        "p95": float(np.percentile(finite_max, 95)) if finite_max.size else float("nan"),
        "p99": float(np.percentile(finite_max, 99)) if finite_max.size else float("nan"),
        "max": float(finite_max.max()) if finite_max.size else float("nan"),
    }

    # Subjects with at least one edge above threshold.
    n_subjects_with_high = int(sum(1 for r in per_subj_top if r["n_high_attention_edges"] > 0))

    # Cross-subject frequency of high-attention (CT-pair, edge-type) tuples.
    if high_attn_records:
        df_high = pd.DataFrame(
            high_attn_records,
            columns=["source_ct", "target_ct", "edge_type", "attention", "subject_id"],
        )
        freq = (
            df_high.groupby(["source_ct", "target_ct", "edge_type"], as_index=False)
            .agg(
                n_subjects=("subject_id", "nunique"),
                n_records=("subject_id", "size"),
                mean_attention=("attention", "mean"),
                max_attention=("attention", "max"),
            )
            .sort_values("n_subjects", ascending=False)
        )
        top_freq_edges = freq.head(10).to_dict(orient="records")
    else:
        top_freq_edges = []

    return {
        "config": {
            "threshold": float(threshold),
            "top_k_per_subject": int(top_k),
            "n_cell_types": int(n_ct),
            "n_edge_types": int(n_et),
        },
        "max_attention_distribution": max_dist,
        "n_subjects_with_high_attention": n_subjects_with_high,
        "frac_subjects_with_high_attention": float(n_subjects_with_high) / max(N, 1),
        "top_frequent_high_attention_edges": top_freq_edges,
        "per_subject": per_subj_top,
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

    extracted = extract_per_subject_attention(args, device)
    attn_arr = extracted["attention"]
    subject_ids = extracted["subject_ids"]
    folds = extracted["folds"]
    logger.info(
        "Extracted attention shape=%s for %d subjects across folds %s",
        attn_arr.shape,
        len(subject_ids),
        sorted(set(int(f) for f in folds)),
    )

    npz_path = out_dir / "per_subject_ccc_attention.npz"
    np.savez_compressed(
        npz_path,
        attention=attn_arr,
        subject_ids=subject_ids,
        folds=folds,
        cell_type_order=np.array(CELL_TYPE_ORDER),
        edge_type_order=np.array(ALL_EDGE_TYPES),
    )
    logger.info("Wrote %s", npz_path)

    summary = summarize(
        attn_arr=attn_arr,
        subject_ids=subject_ids,
        folds=folds,
        threshold=float(args.threshold),
        top_k=int(args.top_k),
    )
    summary["provenance"] = {
        "pred_root": str(args.pred_root),
        "config": str(args.config),
        "splits_path": str(args.splits_path),
        "n_folds": int(args.n_folds),
        "threshold": float(args.threshold),
        "top_k_per_subject": int(args.top_k),
        "npz_path": str(npz_path),
    }
    json_path = out_dir / "per_subject_ccc_attention_summary.json"
    json_path.write_text(json.dumps(summary, indent=2, default=_jsonify))
    logger.info("Wrote %s", json_path)

    # Stdout digest.
    md = summary["max_attention_distribution"]
    print()
    print("=== Per-subject CCC heterogeneity (canonical ResDec-MHE) ===")
    print(f"  N subjects                        : {md['n_subjects']}")
    print(f"  Max-edge-attention mean ± std     : {md['mean']:.5f} ± {md['std']:.5f}")
    print(f"  Max-edge-attention p95 / p99 / max: {md['p95']:.5f} / {md['p99']:.5f} / {md['max']:.5f}")
    print(f"  # subjects with edge > {args.threshold:.3f}    : "
          f"{summary['n_subjects_with_high_attention']} "
          f"({100 * summary['frac_subjects_with_high_attention']:.1f}%)")
    print()
    print(f"  Top-{min(3, len(summary['top_frequent_high_attention_edges']))} "
          f"most-frequent high-attention edges:")
    for r in summary["top_frequent_high_attention_edges"][:3]:
        print(f"    {r['source_ct']:>30s} -> {r['target_ct']:<30s} "
              f"({r['edge_type']:<22s}) — n_subjects={r['n_subjects']:>3d}, "
              f"mean={r['mean_attention']:.5f}, max={r['max_attention']:.5f}")
    return 0


def _jsonify(o):
    if isinstance(o, (np.floating, np.integer)):
        return float(o) if isinstance(o, np.floating) else int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.bool_):
        return bool(o)
    raise TypeError(f"Cannot JSON-serialize {type(o)}: {o!r}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Per-subject CCC HGT edge attention extraction")
    p.add_argument("--config", default="configs/resdec_mhe/canonical.yaml")
    p.add_argument("--pred-root", default="outputs/canonical/p5_canonical_seed42")
    p.add_argument("--splits-path", default="outputs/splits.json")
    p.add_argument("--out-dir", default="outputs/canonical/interpretability/ccc")
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument(
        "--threshold", type=float, default=0.01,
        help=("High-attention threshold (default 0.01 ≈ 5x the population-mean "
              "attention; corresponds to >5σ above mean per Finding 9)."),
    )
    p.add_argument("--top-k", type=int, default=5,
                   help="Number of top edges to record per subject.")
    sys.exit(main(p.parse_args()))
