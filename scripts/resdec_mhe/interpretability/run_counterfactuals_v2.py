"""Counterfactuals v2 — fixes the three caveats of v1 (`run_counterfactuals.py`).

Diff vs v1 (`run_counterfactuals.py`):
  1. **Regime-based target** instead of sign-based. Previously
     `target_y = y_init - delta if y_init > 0 else y_init + delta`; now
     `target_y = y_init - delta if regime == "resilient" else y_init + delta`,
     which guarantees the target always sits on the opposite-regime side of
     the canonical residual (v1 occasionally pushed subjects deeper into
     their own regime when y_init's sign disagreed with the residual-quartile
     label).
  2. **PFC-slice-only perturbation.** v1 perturbed all 6 regions ×
     31 CTs × 4,785 genes = 890,010 dims even though 88 % of ROSMAP
     subjects are PFC-only and their non-PFC regions are architectural
     zero-inputs. v2 perturbs only the PFC slice (1 × 31 × 4,785 =
     148,335 dims); the other 5 regions retain their original
     (usually zero) values, and their gradient is never taken.
  3. **Stronger search budget.** Default `max_steps` raised from 500 →
     2000 and `lambda_dist` lowered from 0.1 → 0.01 so more subjects
     can cross the target tolerance (v1 had 0 / 20 success rate).

The v1 script is kept on disk unchanged for historical reference.
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
from src.data.constants import PFC_REGION_IDX
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


def _build_pfc_only_closures(
    model: ResDecLightningModule,
    template_batch: dict,
    device: torch.device,
):
    """Build (f, grad_f) closures for PFC-slice-only perturbation.

    The perturbation vector ``x`` has shape (n_ct * n_gene,) and, when
    reshaped to (1, n_ct, n_gene), is injected into region PFC of a copy
    of the subject's ``region_pseudobulk`` tensor. Other regions retain
    whatever values they had in the template batch (usually zero for
    PFC-only subjects, real data for multi-region subjects). Gradient is
    taken only with respect to the PFC slice.
    """
    if "region_pseudobulk" not in template_batch:
        raise KeyError(
            "template_batch is missing 'region_pseudobulk'; v2 requires "
            "the multi-region format (datamodule always provides this)"
        )
    orig_rp = template_batch["region_pseudobulk"].clone()
    _, n_regions, n_ct, n_gene = orig_rp.shape
    pfc_shape = (1, n_ct, n_gene)
    n_features = int(np.prod(pfc_shape))

    def _forward_with_x(x_np: np.ndarray, requires_grad: bool):
        xt = torch.tensor(
            x_np, dtype=torch.float32, device=device,
        ).reshape(pfc_shape)
        if requires_grad:
            xt.requires_grad_(True)
        rp = orig_rp.clone()
        # NOTE: we do NOT replace rp[:, PFC_REGION_IDX, :, :] via in-place
        # assignment, because that breaks autograd on xt. Instead, build a
        # new tensor via concat / scatter.
        rp_pfc_slice = rp[:, :PFC_REGION_IDX, :, :]  # regions before PFC (empty if PFC_REGION_IDX=0)
        rp_post_slice = rp[:, PFC_REGION_IDX + 1:, :, :]  # regions after PFC
        new_rp = torch.cat([rp_pfc_slice, xt.unsqueeze(1), rp_post_slice], dim=1)
        batch = dict(template_batch)
        batch["region_pseudobulk"] = new_rp
        out = model(batch)
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

    return f, grad_f, n_features, pfc_shape


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
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--out-dir",
        default="outputs/redesign/interpretability/counterfactuals_v2",
    )
    p.add_argument("--n-resilient", type=int, default=10)
    p.add_argument("--n-vulnerable", type=int, default=10)
    p.add_argument("--max-steps", type=int, default=2000,
                   help="v2 default 2000 vs v1 500 — stronger search.")
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--lambda-dist", type=float, default=0.01,
                   help="v2 default 0.01 vs v1 0.1 — weaker regularization.")
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
    dm.batch_size = 1  # per-subject perturbation; avoids flattened-edge-tensor slicing issues.

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

    results: list[dict] = []
    t0 = time.time()
    n_features = None
    for idx, (sid, template_batch) in enumerate(subject_batches):
        if "region_pseudobulk" not in template_batch or template_batch["region_pseudobulk"] is None:
            logger.warning("subject %s: no region_pseudobulk; skipping", sid)
            continue
        f, grad_f, n_features, pfc_shape = _build_pfc_only_closures(
            model, template_batch, device,
        )
        # Initial x is the subject's PFC slice flattened.
        x_init = (
            template_batch["region_pseudobulk"][:, PFC_REGION_IDX, :, :]
            .detach().cpu().numpy().reshape(-1).astype(np.float64)
        )
        y_init = f(x_init)
        regime = regime_map[sid]
        # Regime-based target (two modes):
        #   relative — push y_init by ±target_delta in opposite-regime direction
        #   absolute — force the prediction to ±target_delta regardless of y_init
        if args.target_mode == "absolute":
            target_y = (
                -args.target_delta if regime == "resilient"
                else args.target_delta
            )
        else:  # relative
            target_y = (
                y_init - args.target_delta if regime == "resilient"
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
            "regime": regime,
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
            "steps=%d, L2=%.3g, success=%s, t=%.1fs",
            idx + 1, len(subject_batches), sid, regime,
            cf.y_init, cf.y_cf, cf.target_y, cf.n_steps_used,
            cf.l2_distance, cf.success, results[-1]["elapsed_s"],
        )

    summary = {
        "version": "v2",
        "target_mode": args.target_mode,
        "fold": args.fold,
        "ckpt": str(ckpt_path),
        "n_features_per_subject": n_features,
        "pfc_shape": list(pfc_shape) if pfc_shape else None,
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
        "v2_changes_vs_v1": [
            "target-logic: sign-based → regime-based",
            "perturbation-space: full region_pseudobulk (890,010 dims) → PFC slice only (148,335 dims)",
            "search budget: max_steps 500 → 2000, lambda_dist 0.1 → 0.01",
            f"target-mode: {args.target_mode} (relative = y_init-delta push; absolute = target ±delta)",
        ],
    }
    out_path = out_dir / f"counterfactuals_fold{args.fold}.json"
    out_path.write_text(json.dumps(summary, indent=2))
    logger.info("wrote %s (%.1f min total)", out_path, summary["elapsed_min"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
