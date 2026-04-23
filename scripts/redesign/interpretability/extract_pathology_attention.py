"""Extract per-subject PathologyAttention weights from canonical ResDec-H3 ckpts.

For each fold, loads the max-R² ``best-*.ckpt``, runs forward over the val
subjects, and captures ``enc_out["attention_weights"]`` of shape
``[B, n_heads, n_cell_types]`` from the encoder's PathologyStratifiedAttention
module. Stacks across all 5 folds → 516 subjects × n_heads × 31 cell types.

Outputs (default ``outputs/redesign/interpretability/``):
  - pathology_attention_per_subject.npz  — keys: subject_ids [N], attention [N, H, C], fold [N]
  - pathology_attention_summary.json     — mean / std attention per (head, cell_type),
                                            + top-5 cell-types by mean attention.

Usage
-----
    PYTHONPATH=<worktree-root> \\
    CUDA_VISIBLE_DEVICES=0 \\
    uv run python scripts/redesign/interpretability/extract_pathology_attention.py \\
        --pred-root outputs/redesign/p5_canonical_seed42 \\
        --out-dir outputs/redesign/interpretability

Arguments
---------
    --config <path>            Phase YAML to merge on top of configs/default.yaml
                               (default: canonical p5_phase2_residual.yaml).
    --pred-root <path>         Per-fold output dir holding fold{0..4}/checkpoints/best-*.ckpt.
    --splits-path <path>       Splits JSON (default: outputs/splits.json).
    --out-dir <path>           Output directory (created if missing).
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

_WORKTREE_ROOT = Path(__file__).resolve().parents[3]
if not (_WORKTREE_ROOT / "src").is_dir():
    raise RuntimeError(
        f"sys.path bootstrap failed: {_WORKTREE_ROOT}/src not found; "
        "set PYTHONPATH=<worktree-root>."
    )
sys.path.insert(0, str(_WORKTREE_ROOT))

from src.data.constants import CELL_TYPE_ORDER
from src.data.datamodule import CognitiveResilienceDataModule
from src.data.splits import load_splits
from src.training.resdec_lightning_module import ResDecLightningModule

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


def extract_one_fold(args: argparse.Namespace, fold: int,
                     device: torch.device) -> dict:
    """Load fold's best ckpt, forward over val, return per-subject attention."""
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
    dm.setup(stage="fit")  # creates _val_ds (validate stage alone won't)

    model = ResDecLightningModule.load_from_checkpoint(
        str(ckpt_path), config=cfg, map_location="cpu",
    ).to(device).eval()

    sids_all: list[str] = []
    attn_chunks: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in dm.val_dataloader():
            sids = list(batch["subject_ids"])
            batch_d = {
                k: v.to(device) if torch.is_tensor(v) else v
                for k, v in batch.items()
            }
            out = model.forward(batch_d)
            attn = out.get("attention_weights")
            if attn is None:
                raise RuntimeError(
                    "Encoder forward did not return 'attention_weights'. The "
                    "PathologyStratifiedAttention module should populate this "
                    "key in CognitiveResilienceModel.forward — verify that "
                    "the encoder build path includes pathology attention."
                )
            sids_all.extend(sids)
            attn_chunks.append(attn.detach().cpu())

    attn_tensor = torch.cat(attn_chunks).float().numpy()
    return {
        "subject_ids": np.array(sids_all, dtype=object),
        "attention": attn_tensor,
        "fold": np.full(len(sids_all), fold, dtype=np.int32),
    }


def main(args: argparse.Namespace) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    sids_per: list[np.ndarray] = []
    attn_per: list[np.ndarray] = []
    fold_per: list[np.ndarray] = []
    for f in range(5):
        d = extract_one_fold(args, f, device)
        sids_per.append(d["subject_ids"])
        attn_per.append(d["attention"])
        fold_per.append(d["fold"])
        logger.info("fold %d: extracted attention shape %s", f, d["attention"].shape)

    sids = np.concatenate(sids_per)
    attn = np.concatenate(attn_per, axis=0)
    folds = np.concatenate(fold_per)
    logger.info("Total: %d subjects; attention shape %s", len(sids), attn.shape)

    out_npz = out_dir / "pathology_attention_per_subject.npz"
    np.savez(out_npz, subject_ids=sids, attention=attn, fold=folds)
    logger.info("Wrote %s", out_npz)

    # Summary stats: mean/std per (head, cell_type) + top-5 cell types by
    # mean attention averaged across heads.
    n_heads = int(attn.shape[1])
    n_ct = int(attn.shape[2])
    mean_per = attn.mean(axis=0)  # [H, C]
    std_per = attn.std(axis=0)

    cell_type_names = list(CELL_TYPE_ORDER)
    if len(cell_type_names) < n_ct:
        # Pad with placeholders if the constants list is shorter than the model.
        cell_type_names = cell_type_names + [f"ct_{c}" for c in range(len(cell_type_names), n_ct)]

    summary = {
        "n_subjects": int(len(sids)),
        "n_heads": n_heads,
        "n_cell_types": n_ct,
        "cell_type_names_used": cell_type_names[:n_ct],
        "per_head_per_cell_type": [
            {
                "head": h,
                "cell_type": cell_type_names[c],
                "mean_attention": float(mean_per[h, c]),
                "std_attention": float(std_per[h, c]),
            }
            for h in range(n_heads) for c in range(n_ct)
        ],
        "top_5_cell_types_avg_over_heads": [
            {
                "cell_type": cell_type_names[c],
                "mean_attention_avg_heads": float(mean_per[:, c].mean()),
                "std_attention_avg_heads": float(std_per[:, c].mean()),
            }
            for c in np.argsort(-mean_per.mean(axis=0))[:5]
        ],
    }
    summary_path = out_dir / "pathology_attention_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    logger.info("Wrote %s", summary_path)

    print("\n=== Top-5 cell types by mean attention (averaged over heads) ===")
    for entry in summary["top_5_cell_types_avg_over_heads"]:
        print(f"  {entry['cell_type']:<40s}  mean={entry['mean_attention_avg_heads']:.4f}  "
              f"std={entry['std_attention_avg_heads']:.4f}")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Extract PathologyAttention weights.")
    p.add_argument("--config", default="configs/redesign/p5_phase2_residual.yaml",
                   help="Phase YAML merged on top of configs/default.yaml.")
    p.add_argument("--pred-root", default="outputs/redesign/p5_canonical_seed42",
                   help="Per-fold output dir with fold{0..4}/checkpoints/best-*.ckpt.")
    p.add_argument("--splits-path", default="outputs/splits.json")
    p.add_argument("--out-dir", default="outputs/redesign/interpretability")
    sys.exit(main(p.parse_args()))
