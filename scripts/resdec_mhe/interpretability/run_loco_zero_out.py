"""Leave-One-Cell-type-Out (LOCO) ablation via zero-out at inference.

For each cell type ``ct`` ∈ ``[0, n_cell_types)``, zero out that cell type's
pseudobulk in the **val** input of each fold, run the trained canonical
model forward, compose with the (fixed) TabPFN outer prediction, and
compute the per-fold + aggregate 5-fold R². The delta between canonical
5-fold R² and the LOCO R² is the per-CT sensitivity.

This is the *inference-only* LOCO (no retraining). Checkpoints
themselves are untouched — only the input cell type tensor is zeroed
per forward pass.

Outputs:
  ``<out-dir>/loco_per_celltype.{csv,json}`` — per-CT sensitivity table.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import pearsonr

_WORKTREE_ROOT = Path(__file__).resolve().parents[3]
if str(_WORKTREE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKTREE_ROOT))

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
        raise FileNotFoundError(f"No best-*.ckpt in {ckpt_dir}")
    return best[0]


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_WORKTREE_ROOT,
        ).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _zero_out_and_predict(
    model: ResDecLightningModule,
    val_batches: list[dict],
    ct_to_zero: int | None,
    tabpfn_outer_map: dict[str, float],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run forward over val_batches with cell-type ct_to_zero zeroed.

    If ``ct_to_zero is None``, runs canonical (no zero). Returns
    (subject_ids, composite_preds, true_y).
    """
    sids_all: list[str] = []
    comp_all: list[float] = []
    true_all: list[float] = []
    with torch.no_grad():
        for batch in val_batches:
            batch_d = {
                k: v.to(device) if torch.is_tensor(v) else v
                for k, v in batch.items()
            }
            if ct_to_zero is not None:
                for key in ("pseudobulk", "region_pseudobulk"):
                    v = batch_d.get(key)
                    if v is None or not torch.is_tensor(v):
                        continue
                    v_mod = v.clone()
                    if key == "pseudobulk":
                        # shape [B, n_cell_types, n_genes]
                        v_mod[:, ct_to_zero, :] = 0.0
                    else:  # region_pseudobulk: [B, n_regions, n_cell_types, n_genes]
                        v_mod[:, :, ct_to_zero, :] = 0.0
                    batch_d[key] = v_mod
            out = model(batch_d)
            residual = out["prediction"].detach().cpu().numpy().reshape(-1)
            for i, sid in enumerate(batch["subject_ids"]):
                sid_str = str(sid)
                ytab = tabpfn_outer_map.get(sid_str, np.nan)
                composite = float(residual[i]) + float(ytab)
                sids_all.append(sid_str)
                comp_all.append(composite)
                true_all.append(
                    float(batch_d["cognition"][i].item())
                    if "cognition" in batch_d else np.nan,
                )
    return (
        np.asarray(sids_all, dtype=object),
        np.asarray(comp_all, dtype=np.float64),
        np.asarray(true_all, dtype=np.float64),
    )


def _load_tabpfn_outer_map(tabpfn_dir: Path, fold: int) -> dict[str, float]:
    path = tabpfn_dir / f"tabpfn_outer_fold{fold}.npz"
    d = np.load(path, allow_pickle=True)
    return {
        str(s): float(v) for s, v in zip(d["val_subject_ids"], d["y_tabpfn"])
    }


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--config", default="configs/resdec_mhe/canonical.yaml")
    p.add_argument(
        "--canonical-dir", default="outputs/redesign/p5_canonical_seed42",
    )
    p.add_argument("--tabpfn-dir", default="data/redesign")
    p.add_argument("--splits-path", default="outputs/splits.json")
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--n-cell-types", type=int, default=31)
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--out-dir",
        default="outputs/redesign/interpretability/loco_zero_out",
    )
    p.add_argument(
        "--cell-type-names-source",
        default="outputs/redesign/interpretability/captum_ig/"
        "composite_attribution_summary.json",
    )
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    cfg = OmegaConf.merge(
        OmegaConf.load("configs/default.yaml"),
        OmegaConf.load(args.config),
    )
    OmegaConf.set_struct(cfg, False)
    cfg.model.head.type = "deterministic"

    # Per-fold canonical + LOCO R² tables.
    per_ct_rows: list[dict] = []
    canonical_per_fold: list[float] = []
    loco_per_ct_fold: dict[int, list[float]] = {ct: [] for ct in range(args.n_cell_types)}

    t_start = time.time()
    for fold in range(args.n_folds):
        fold_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        OmegaConf.set_struct(fold_cfg, False)
        fold_cfg.data.fold = fold

        fold_dir = Path(args.canonical_dir) / f"fold{fold}"
        ckpt_path = _pick_max_r2_ckpt(fold_dir / "checkpoints")
        logger.info("fold %d: loading %s", fold, ckpt_path.name)

        splits = load_splits(str(args.splits_path))
        metadata = pd.read_csv(Path(fold_cfg.data.metadata_path) / "metadata.csv")
        dm = CognitiveResilienceDataModule(
            config=fold_cfg, metadata=metadata, splits=splits,
            fold_idx=fold,
            precomputed_dir=fold_cfg.data.precomputed_dir,
            adata=None,
        )
        dm.setup(stage="fit")

        model = ResDecLightningModule.load_from_checkpoint(
            str(ckpt_path), config=fold_cfg, map_location="cpu",
        ).to(device).eval()

        # Collect val batches once (avoid re-loading per CT).
        val_batches: list[dict] = []
        for batch in dm.val_dataloader():
            val_batches.append(batch)
        tabpfn_map = _load_tabpfn_outer_map(
            Path(args.tabpfn_dir), fold,
        )

        # Canonical (no zero).
        sids, comp, true_y = _zero_out_and_predict(
            model, val_batches, None, tabpfn_map, device,
        )
        r2_canon = float(r2_score(true_y, comp))
        canonical_per_fold.append(r2_canon)
        logger.info("fold %d canonical R² = %+.4f", fold, r2_canon)

        # Per-CT LOCO.
        for ct in range(args.n_cell_types):
            _, comp_ct, true_ct = _zero_out_and_predict(
                model, val_batches, ct, tabpfn_map, device,
            )
            r2_ct = float(r2_score(true_ct, comp_ct))
            loco_per_ct_fold[ct].append(r2_ct)
        logger.info(
            "fold %d: LOCO done over %d CTs (ΔR² range %+.4f…%+.4f)",
            fold, args.n_cell_types,
            min(loco_per_ct_fold[c][fold] - r2_canon for c in range(args.n_cell_types)),
            max(loco_per_ct_fold[c][fold] - r2_canon for c in range(args.n_cell_types)),
        )
        del model

    # Cell type names (best effort).
    ct_names: list[str] = [f"CT_{i}" for i in range(args.n_cell_types)]
    src = Path(args.cell_type_names_source)
    if src.exists():
        s = json.loads(src.read_text())
        raw = (
            s.get("cell_types_ranked_by_total_attribution")
            or s.get("cell_types")
        )
        if isinstance(raw, list) and raw and isinstance(raw[0], dict):
            # Ranked order — do NOT axis-align; keep placeholders per-index.
            pass
        elif isinstance(raw, list):
            ct_names = list(raw)[:args.n_cell_types]

    canon_mean = float(np.mean(canonical_per_fold))
    for ct in range(args.n_cell_types):
        r2s = loco_per_ct_fold[ct]
        mean_r2 = float(np.mean(r2s))
        std_r2 = float(np.std(r2s, ddof=1))
        per_ct_rows.append({
            "cell_type_index": ct,
            "cell_type": ct_names[ct],
            "loco_mean_r2": mean_r2,
            "loco_std_r2": std_r2,
            "per_fold_r2": r2s,
            "delta_r2_vs_canonical": mean_r2 - canon_mean,
        })

    df = pd.DataFrame(per_ct_rows)
    df = df.sort_values("delta_r2_vs_canonical")
    (out_dir / "loco_per_celltype.csv").write_text(df.to_csv(index=False))
    provenance = {
        "canonical_5fold_r2": canonical_per_fold,
        "canonical_mean_r2": canon_mean,
        "canonical_std_r2": float(np.std(canonical_per_fold, ddof=1)),
        "n_folds": args.n_folds,
        "n_cell_types": args.n_cell_types,
        "device": str(device),
        "elapsed_min": round((time.time() - t_start) / 60, 2),
        "git_commit": _git_sha(),
    }
    (out_dir / "loco_per_celltype.json").write_text(
        json.dumps({"per_cell_type": per_ct_rows, "provenance": provenance}, indent=2)
    )
    logger.info(
        "wrote %s (%.1f min; canonical mean R² = %+.4f)",
        out_dir / "loco_per_celltype.csv",
        provenance["elapsed_min"], canon_mean,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
