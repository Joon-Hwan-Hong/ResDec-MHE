"""Phase 3 Task 3.2 sanity: stage-wise corrcoef on val set.

Loads the best-by-val/r2 Phase-3 ResDec-H3 checkpoint for a single fold,
runs a forward pass over the validation set, gathers per-subject
stage scalars ``f̂_1, ..., f̂_N`` (N = ``cfg.model.resdec_head.n_stages``),
and prints pairwise ``corrcoef(f̂_i, f̂_j)`` for all present stage pairs.

Plan target (when N >= 2): both ``corrcoef(f̂_1, f̂_2)`` and
``corrcoef(f̂_1, f̂_3)`` must be < 0.3 (stages learning distinct signal).
When ``n_stages == 1`` the check is a no-op (only stage_1 exists, so there
is no pairwise corrcoef to compute) and the script exits cleanly with a
message to that effect.

Usage
-----
    CUDA_VISIBLE_DEVICES=0 \\
    PYTHONPATH=<worktree-root> \\
    uv run python scripts/redesign/phase3_sanity_corrcoef.py \\
        --config configs/redesign/p5_phase2_residual.yaml \\
        --fold 0 \\
        --output-dir outputs/redesign/p5_phase3

Arguments
---------
    --config <path>            phase YAML merged on top of configs/default.yaml.
    --fold <int>               fold index (required).
    --output-dir <path>        directory holding fold{N}/checkpoints/best-*.ckpt.
    --splits-path <path>       splits JSON (default: outputs/splits.json).
    --precomputed-dir <path>   Override cfg.data.precomputed_dir (optional;
                               defaults to the value from the merged config).
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

# Make the script standalone-runnable: ensure the worktree root is on sys.path
# so `src.*` imports resolve without the caller having to set PYTHONPATH.
# Anchored at parents[2] (i.e. scripts/redesign/<this_file> → worktree_root/).
# Mirrors the pattern used by scripts/redesign/run_tabpfn_attribution.py.
_WORKTREE_ROOT = Path(__file__).resolve().parents[2]
if not (_WORKTREE_ROOT / "src").is_dir():
    raise RuntimeError(
        f"sys.path bootstrap failed: {_WORKTREE_ROOT}/src not found; "
        "set PYTHONPATH=<worktree-root>."
    )
sys.path.insert(0, str(_WORKTREE_ROOT))

from src.data.datamodule import CognitiveResilienceDataModule
from src.data.splits import load_splits
from src.models.resdec_head.resdec_h3_head import DEFAULT_N_STAGES
from src.training.resdec_lightning_module import ResDecLightningModule

logger = logging.getLogger(__name__)
_BEST_CKPT_RE = re.compile(r"^best-(\d+)-(\d+\.\d+)\.ckpt$")


def _pick_max_r2_ckpt(ckpt_dir: Path) -> tuple[Path, int, float]:
    best: tuple[Path, int, float] | None = None
    for p in ckpt_dir.glob("best-*.ckpt"):
        m = _BEST_CKPT_RE.match(p.name)
        if not m:
            continue
        epoch, r2 = int(m.group(1)), float(m.group(2))
        if best is None or r2 > best[2]:
            best = (p, epoch, r2)
    if best is None:
        raise FileNotFoundError(f"No best-*.ckpt files in {ckpt_dir}")
    return best


def main(args: argparse.Namespace) -> int:
    logging.basicConfig(level=logging.INFO)
    default_cfg = OmegaConf.load("configs/default.yaml")
    cfg = OmegaConf.merge(default_cfg, OmegaConf.load(args.config))
    OmegaConf.set_struct(cfg, False)
    cfg.model.head.type = "deterministic"
    cfg.data.fold = int(args.fold)

    # Read n_stages from the merged config (canonical default is 1).
    resdec_cfg = cfg.model.get("resdec_head", {}) or {}
    n_stages = int(resdec_cfg.get("n_stages", DEFAULT_N_STAGES))

    if n_stages < 2:
        logger.info(
            "n_stages=%d: corrcoef sanity check requires n_stages>=2 "
            "(only stage_1 is present; no pairwise correlation to compute). "
            "Exiting cleanly.",
            n_stages,
        )
        print(json.dumps({
            "fold": args.fold,
            "n_stages": n_stages,
            "status": "skipped",
            "reason": "n_stages < 2: no pairwise corrcoef to compute",
        }, indent=2))
        return 0

    fold_dir = Path(args.output_dir) / f"fold{args.fold}"
    ckpt_path, epoch, ckpt_r2 = _pick_max_r2_ckpt(fold_dir / "checkpoints")
    logger.info("Loading %s (epoch=%d, filename-R²=%.4f)", ckpt_path.name, epoch, ckpt_r2)

    splits = load_splits(str(args.splits_path))
    metadata = pd.read_csv(Path(cfg.data.metadata_path) / "metadata.csv")
    dm = CognitiveResilienceDataModule(
        config=cfg, metadata=metadata, splits=splits,
        fold_idx=args.fold,
        precomputed_dir=args.precomputed_dir or cfg.data.precomputed_dir,
        adata=None,
    )
    dm.setup(stage="fit")

    model = ResDecLightningModule.load_from_checkpoint(
        str(ckpt_path), config=cfg, map_location="cpu",
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    # Collect per-stage scalars over all val batches, only for present stages.
    stage_chunks: dict[int, list[torch.Tensor]] = {k: [] for k in range(1, n_stages + 1)}
    with torch.no_grad():
        for batch in dm.val_dataloader():
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            out = model.forward(batch)
            for k in range(1, n_stages + 1):
                stage_chunks[k].append(out[f"stage_{k}"].detach().cpu())

    stage_arrays: dict[int, np.ndarray] = {
        k: torch.cat(chunks).float().numpy() for k, chunks in stage_chunks.items()
    }
    n_val = len(stage_arrays[1])

    # Per-stage summary stats.
    per_stage_stats: dict[str, float] = {}
    for k, arr in stage_arrays.items():
        per_stage_stats[f"stage_{k}_mean"] = float(arr.mean())
        per_stage_stats[f"stage_{k}_std"] = float(arr.std())

    # Pairwise corrcoefs for every (i, j) with i < j ∈ [1, n_stages].
    pairwise: dict[str, float] = {}
    for i in range(1, n_stages + 1):
        for j in range(i + 1, n_stages + 1):
            r_ij = float(np.corrcoef(stage_arrays[i], stage_arrays[j])[0, 1])
            pairwise[f"corrcoef_f{i}_f{j}"] = r_ij

    # Pass flags: Plan targets corrcoef(f1, f2) < 0.3 and corrcoef(f1, f3) < 0.3.
    # Only emit keys whose underlying pair exists (so n_stages=2 omits f1_f3).
    pass_flags: dict[str, bool] = {}
    for pair_name, threshold_name in (
        ("corrcoef_f1_f2", "pass_f1_f2_lt_0.3"),
        ("corrcoef_f1_f3", "pass_f1_f3_lt_0.3"),
    ):
        if pair_name in pairwise:
            pass_flags[threshold_name] = pairwise[pair_name] < 0.3

    # Human-readable log: list all pairwise corrcoefs on one line.
    log_pairs = ", ".join(f"{k}={v:.4f}" for k, v in pairwise.items())
    logger.info("n_val=%d, %s", n_val, log_pairs)

    result: dict = {
        "fold": args.fold,
        "n_stages": n_stages,
        "ckpt": ckpt_path.name,
        "n_val": n_val,
        **pairwise,
        **per_stage_stats,
        **pass_flags,
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Phase 3 stage-wise corrcoef sanity check.")
    p.add_argument("--config", default="configs/redesign/p5_phase2_residual.yaml")
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--output-dir", default="outputs/redesign/p5_phase3")
    p.add_argument("--splits-path", default="outputs/splits.json")
    p.add_argument("--precomputed-dir", default=None)
    sys.exit(main(p.parse_args()))
