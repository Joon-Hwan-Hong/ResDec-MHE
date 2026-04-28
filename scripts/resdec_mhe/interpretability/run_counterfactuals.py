"""Per-subject counterfactual resilience explanations (Wachter et al. 2017).

For each sampled subject, find the smallest perturbation of their per-(cell type,
gene) pseudobulk that drives the ResDec-MHE composite prediction toward a
target value (flip toward the opposite resilience regime). Uses the Wachter
quadratic loss ``(f(x) - y_target)^2 + lambda * ||x - x_init||^2``,
gradient-descent optimised by ``find_counterfactual`` from
``src.analysis.counterfactual_resilience``.

Subject selection (default): top-N resilient + top-N vulnerable by canonical
residual, covering both regimes.

Outputs JSON with per-subject {x_init, x_cf (sparse top-K only), y_init,
y_cf, target_y, success, steps, l2_distance, top_k_features}. The full
x_cf tensor is NOT saved (148K features × N subjects is large); instead
we keep the top-K |x_cf - x_init| feature indices + values per subject.
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

_WORKTREE_ROOT = Path(__file__).resolve().parents[3]
if str(_WORKTREE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKTREE_ROOT))

from src.analysis.counterfactual_resilience import find_counterfactual
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
        raise FileNotFoundError(f"No best-*.ckpt in {ckpt_dir}")
    return best[0]


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_WORKTREE_ROOT,
        ).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _build_subject_closures(
    model: ResDecLightningModule,
    template_batch: dict,
    pseudobulk_key: str,
    device: torch.device,
):
    """Build (f, grad_f) closures for one subject's pseudobulk perturbation.

    ``template_batch`` is the batch dict for ONE subject, already on
    ``device``. ``pseudobulk_key`` is either 'pseudobulk' (single-region
    format) or 'region_pseudobulk' (multi-region). The closure accepts a
    flat numpy vector ``x`` of length ``prod(shape)`` and writes it into
    ``template_batch[pseudobulk_key]`` before calling model.forward.
    """
    target_shape = template_batch[pseudobulk_key].shape
    n_features = int(np.prod(target_shape))

    def _forward_with_x(x_np: np.ndarray, requires_grad: bool):
        xt = torch.tensor(
            x_np, dtype=torch.float32, device=device,
        ).reshape(target_shape)
        if requires_grad:
            xt.requires_grad_(True)
        batch = dict(template_batch)
        batch[pseudobulk_key] = xt
        out = model(batch)
        # ResDecLightningModule.forward returns {'prediction': [B], ...}.
        # This is the encoder+head RESIDUAL prediction; the composite is
        # ŷ_tabpfn (cached, fixed per subject) + prediction. For CF search
        # we target the residual, since the TabPFN base is not a function
        # of the perturbed pseudobulk.
        y = out["prediction"].squeeze()
        return y, xt

    def f(x: np.ndarray) -> float:
        with torch.no_grad():
            y, _ = _forward_with_x(x, requires_grad=False)
            return float(y.detach().cpu().numpy())

    def grad_f(x: np.ndarray) -> np.ndarray:
        y, xt = _forward_with_x(x, requires_grad=True)
        y.backward()
        g = xt.grad.reshape(-1).detach().cpu().numpy().astype(np.float64)
        return g

    return f, grad_f, n_features


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--config", default="configs/resdec_mhe/canonical.yaml")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument(
        "--canonical-dir", default="outputs/redesign/p5_canonical_seed42",
    )
    p.add_argument("--splits-path", default="outputs/splits.json")
    p.add_argument(
        "--residual-csv",
        default="outputs/redesign/interpretability/residual_per_subject.csv",
    )
    p.add_argument("--device", default="cpu")
    p.add_argument(
        "--out-dir",
        default="outputs/redesign/interpretability/counterfactuals",
    )
    p.add_argument("--n-resilient", type=int, default=10,
                   help="Number of top-resilient subjects to perturb.")
    p.add_argument("--n-vulnerable", type=int, default=10,
                   help="Number of top-vulnerable subjects to perturb.")
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--lambda-dist", type=float, default=0.1)
    p.add_argument("--top-k", type=int, default=50,
                   help="Number of top-perturbed features to save per subject.")
    p.add_argument("--target-delta", type=float, default=0.5,
                   help="Target y offset from initial prediction (flip magnitude).")
    p.add_argument("--seed", type=int, default=42)
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
    cfg.data.fold = int(args.fold)

    fold_dir = Path(args.canonical_dir) / f"fold{args.fold}"
    ckpt_path = _pick_max_r2_ckpt(fold_dir / "checkpoints")
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
    # Force batch_size=1 for per-subject pseudobulk perturbation. ccc_edge_*
    # tensors are flattened across the batch (not per-subject), so manual
    # slicing breaks the HGT edge-index invariant — single-subject batches
    # sidestep the issue entirely.
    dm.batch_size = 1

    model = ResDecLightningModule.load_from_checkpoint(
        str(ckpt_path), config=cfg, map_location="cpu",
    ).to(device).eval()

    # Residuals for subject selection.
    res_df = pd.read_csv(args.residual_csv)
    id_col = (
        "ROSMAP_IndividualID" if "ROSMAP_IndividualID" in res_df.columns
        else res_df.columns[0]
    )
    res_df = res_df.rename(columns={id_col: "subject_id"})
    res_df["subject_id"] = res_df["subject_id"].astype(str)
    # Select subjects from THIS fold's val split.
    val_subject_ids: set[str] = set()
    for batch in dm.val_dataloader():
        val_subject_ids.update(str(s) for s in batch["subject_ids"])
    val_res = res_df[res_df["subject_id"].isin(val_subject_ids)].copy()
    val_res = val_res[np.isfinite(val_res["residual"])]
    top_res = val_res.nlargest(args.n_resilient, "residual")["subject_id"].tolist()
    top_vul = val_res.nsmallest(args.n_vulnerable, "residual")["subject_id"].tolist()
    target_subjects = set(top_res) | set(top_vul)
    logger.info(
        "selected %d resilient + %d vulnerable val-fold subjects for CF search",
        len(top_res), len(top_vul),
    )

    # Iterate val dataloader at batch_size=1; collect per-subject batches.
    results: list[dict] = []
    t0 = time.time()
    subject_batches: list[tuple[str, dict]] = []
    for batch in dm.val_dataloader():
        sids = list(batch["subject_ids"])
        assert len(sids) == 1, f"expected batch_size=1; got {len(sids)}"
        sid_str = str(sids[0])
        if sid_str not in target_subjects:
            continue
        per = {
            k: (v.to(device) if torch.is_tensor(v) else v)
            for k, v in batch.items()
        }
        subject_batches.append((sid_str, per))

    for idx, (sid, template_batch) in enumerate(subject_batches):
        if "region_pseudobulk" in template_batch and template_batch["region_pseudobulk"] is not None:
            pseudobulk_key = "region_pseudobulk"
        elif "pseudobulk" in template_batch and template_batch["pseudobulk"] is not None:
            pseudobulk_key = "pseudobulk"
        else:
            logger.warning("subject %s: no pseudobulk key; skipping", sid)
            continue
        f, grad_f, n_features = _build_subject_closures(
            model, template_batch, pseudobulk_key, device,
        )
        x_init = (
            template_batch[pseudobulk_key].detach().cpu().numpy()
            .reshape(-1).astype(np.float64)
        )
        y_init = f(x_init)
        # Flip target: if y_init > 0 push toward -delta; else toward +delta.
        target_y = (
            y_init - args.target_delta if y_init > 0
            else y_init + args.target_delta
        )
        t_s = time.time()
        cf = find_counterfactual(
            f, grad_f, x_init, target_y,
            lr=args.lr, lambda_dist=args.lambda_dist,
            max_steps=args.max_steps, seed=args.seed,
        )
        delta = np.abs(cf.x_cf - cf.x_init)
        top_k_idx = np.argsort(delta)[::-1][:args.top_k]
        top_k = [
            {
                "feature_idx": int(i),
                "abs_delta": float(delta[i]),
                "x_init": float(cf.x_init[i]),
                "x_cf": float(cf.x_cf[i]),
            }
            for i in top_k_idx
        ]
        results.append({
            "subject_id": sid,
            "regime": "resilient" if sid in set(top_res) else "vulnerable",
            "y_init": cf.y_init,
            "y_cf": cf.y_cf,
            "target_y": cf.target_y,
            "success": cf.success,
            "n_steps_used": cf.n_steps_used,
            "l2_distance": cf.l2_distance,
            "elapsed_s": round(time.time() - t_s, 1),
            "top_k_features": top_k,
        })
        logger.info(
            "[%d/%d] %s (%s): y=%+.3f → %+.3f (target %+.3f), "
            "steps=%d, L2=%.3g, t=%.1fs",
            idx + 1, len(subject_batches), sid,
            results[-1]["regime"], cf.y_init, cf.y_cf, cf.target_y,
            cf.n_steps_used, cf.l2_distance, results[-1]["elapsed_s"],
        )

    summary = {
        "fold": args.fold,
        "ckpt": str(ckpt_path),
        "n_features_per_subject": int(n_features) if results else None,
        "n_subjects_processed": len(results),
        "n_resilient_targeted": len(top_res),
        "n_vulnerable_targeted": len(top_vul),
        "max_steps": args.max_steps,
        "lr": args.lr,
        "lambda_dist": args.lambda_dist,
        "target_delta": args.target_delta,
        "top_k_per_subject": args.top_k,
        "seed": args.seed,
        "elapsed_min": round((time.time() - t0) / 60, 2),
        "git_commit": _git_sha(),
        "results": results,
    }
    out_path = out_dir / f"counterfactuals_fold{args.fold}.json"
    out_path.write_text(json.dumps(summary, indent=2))
    logger.info("wrote %s (%.1f min total)", out_path, summary["elapsed_min"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
