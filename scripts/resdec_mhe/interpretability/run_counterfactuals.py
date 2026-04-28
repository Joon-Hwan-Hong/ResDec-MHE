"""Per-subject counterfactual explanations (Wachter et al. 2017 Mode-A literal).

Minimizes ``L(x, λ) = ‖x − x_init‖² + λ · (f(x) − y_target)²`` with
adaptive λ doubling:

  * Start λ at ``--lambda-start`` (small → distance dominates).
  * Run inner gradient descent for up to ``--max-steps`` iterations
    with unit-norm L2 gradient clipping for stability at high λ.
  * If target not reached (``|f(x) − target| > --tol``), reset
    ``x ← x_init``, multiply λ by ``--lambda-mult`` (default 2.0),
    and retry.
  * Terminate at success OR when ``λ > --lambda-max``. The result
    records which λ was the converging / closest attempt.

Perturbations are restricted to the PFC slice of ``region_pseudobulk``
(31 cell types × 4,785 genes = 148 K dimensions). The other five
regions are held at their original (typically zero, since most
ROSMAP subjects are PFC-only) values, and the gradient is taken
only with respect to the PFC slice.

Two target modes are supported:
  - ``relative`` — ``y_target = y_init ± target_delta``: each subject
    is asked to move a fixed displacement in the opposite-regime
    direction, giving constant per-subject search difficulty (best
    for top-K aggregation across subjects).
  - ``absolute`` — ``y_target = ± target_delta``: each subject is
    asked to reach a fixed regime-side prediction value regardless
    of starting point, giving variable difficulty.
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

_WORKTREE_ROOT = Path(__file__).resolve().parents[3]
if str(_WORKTREE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKTREE_ROOT))

from src.analysis.counterfactual_resilience import find_counterfactual_mode_a_adaptive
from src.data.constants import PFC_REGION_IDX
from src.data.datamodule import CognitiveResilienceDataModule
from src.data.splits import load_splits
from src.training.resdec_lightning_module import ResDecLightningModule
from src.utils.provenance import git_sha, pick_max_r2_ckpt

logger = logging.getLogger(__name__)


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
            "template_batch is missing 'region_pseudobulk'; this orchestrator "
            "requires the multi-region format (datamodule always provides this)"
        )
    orig_rp = template_batch["region_pseudobulk"]
    _, n_regions, n_ct, n_gene = orig_rp.shape
    pfc_shape = (1, n_ct, n_gene)
    n_features = int(np.prod(pfc_shape))

    # Pre-compute the constant non-PFC region slices once (they never change
    # across GD steps). The naive implementation would clone orig_rp per step
    # (a 3.4 MB GPU memcpy) and re-slice; both are wasted work since these
    # slices are read-only.
    pre_slice = orig_rp[:, :PFC_REGION_IDX, :, :].contiguous()
    post_slice = orig_rp[:, PFC_REGION_IDX + 1:, :, :].contiguous()

    # Pre-allocate the perturbation tensor once with requires_grad=True.
    # Each call zeroes its grad and copies new x_np data into .data; the
    # autograd graph is rebuilt fresh on each forward (since we re-do
    # torch.cat etc.), but the leaf-tensor allocation is paid once.
    xt_buf = torch.zeros(pfc_shape, dtype=torch.float32, device=device,
                         requires_grad=True)

    # Reusable batch dict — only region_pseudobulk gets swapped each call.
    template_batch_copy = dict(template_batch)

    def _forward_with_x(x_np: np.ndarray, requires_grad: bool):
        if requires_grad:
            # Reset grad and copy fresh data into the leaf tensor. Using
            # data.copy_ avoids creating a new graph node for the assignment.
            if xt_buf.grad is not None:
                xt_buf.grad = None
            with torch.no_grad():
                xt_buf.data.copy_(
                    torch.from_numpy(x_np).to(device).reshape(pfc_shape)
                )
            xt = xt_buf
        else:
            # No-grad path: use a fresh non-leaf tensor (cheaper than
            # touching xt_buf which would carry its requires_grad flag).
            xt = torch.from_numpy(x_np).to(
                device=device, dtype=torch.float32,
            ).reshape(pfc_shape)
        new_rp = torch.cat([pre_slice, xt.unsqueeze(1), post_slice], dim=1)
        template_batch_copy["region_pseudobulk"] = new_rp
        out = model(template_batch_copy)
        y = out["prediction"].squeeze()
        return y, xt

    def f(x: np.ndarray) -> float:
        with torch.no_grad():
            y, _ = _forward_with_x(x, requires_grad=False)
            return float(y.detach().cpu().numpy())

    def grad_f(x: np.ndarray) -> np.ndarray:
        y, xt = _forward_with_x(x, requires_grad=True)
        y.backward()
        return xt.grad.reshape(-1).detach().cpu().numpy().astype(np.float64)

    def f_and_grad(x: np.ndarray) -> tuple[float, np.ndarray]:
        """Combined forward + backward; saves one forward pass per GD step."""
        y, xt = _forward_with_x(x, requires_grad=True)
        y.backward()
        return (
            float(y.detach().cpu().numpy()),
            xt.grad.reshape(-1).detach().cpu().numpy().astype(np.float64),
        )

    return f, grad_f, f_and_grad, n_features, pfc_shape


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--config", default="configs/resdec_mhe/canonical.yaml")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument(
        "--canonical-dir", default="outputs/canonical/p5_canonical_seed42",
    )
    p.add_argument("--splits-path", default="outputs/splits.json")
    p.add_argument(
        "--residual-csv",
        default="outputs/canonical/interpretability/residual_per_subject.csv",
    )
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--out-dir",
        default="outputs/canonical/interpretability/counterfactuals",
    )
    p.add_argument("--n-resilient", type=int, default=10)
    p.add_argument("--n-vulnerable", type=int, default=10)
    p.add_argument("--max-steps", type=int, default=1000,
                   help="Inner GD steps per λ attempt.")
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--tol", type=float, default=1e-3,
                   help="Convergence tolerance |f(x) - target| <= tol.")
    p.add_argument("--lambda-start", type=float, default=1e-3,
                   help="Initial λ (small = distance-dominant).")
    p.add_argument("--lambda-max", type=float, default=1e3,
                   help="Terminate search when λ exceeds this value.")
    p.add_argument("--lambda-mult", type=float, default=2.0,
                   help="λ multiplier per adaptive step (Wachter: 2.0).")
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
    p.add_argument("--no-compile", action="store_true",
                   help="Disable torch.compile (use plain eager mode).")
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
    ckpt_path = pick_max_r2_ckpt(fold_dir / "checkpoints")
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
    # Optimization B: torch.compile the model in eval mode. Saves ~1.1-1.3×
    # on forward+backward via op fusion. Numerically identical to ~1e-6.
    # mode="default" instead of "reduce-overhead" because cudagraphs (used
    # by reduce-overhead) is incompatible with FiLM's gamma_proj(metadata)
    # tensor-reuse pattern (errors out at "tensor output of CUDAGraphs has
    # been overwritten by a subsequent run").
    # Skip via --no-compile when debugging.
    if not args.no_compile:
        try:
            model = torch.compile(model, mode="default", fullgraph=False)
            logger.info("torch.compile enabled (mode=default)")
        except Exception as exc:
            logger.warning("torch.compile failed (%s); using uncompiled model", exc)

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
        f, grad_f, f_and_grad, n_features, pfc_shape = _build_pfc_only_closures(
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
        cf = find_counterfactual_mode_a_adaptive(
            f, grad_f, x_init, target_y,
            lr=args.lr, max_steps=args.max_steps, tol=args.tol,
            lambda_start=args.lambda_start, lambda_max=args.lambda_max,
            lambda_mult=args.lambda_mult, seed=args.seed,
            f_and_grad=f_and_grad,
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
            "lambda_used": cf.lambda_used,
            "elapsed_s": round(time.time() - t_s, 1),
            "top_k_features": top_k,
        })
        logger.info(
            "[%d/%d] %s (%s): y=%+.3f → %+.3f (target %+.3f), "
            "steps=%d, L2=%.3g, λ=%.3g, success=%s, t=%.1fs",
            idx + 1, len(subject_batches), sid, regime,
            cf.y_init, cf.y_cf, cf.target_y, cf.n_steps_used,
            cf.l2_distance, cf.lambda_used, cf.success, results[-1]["elapsed_s"],
        )

    summary = {
        "method": "Wachter et al. 2017 Mode-A literal, adaptive λ doubling",
        "loss": "L(x, λ) = ‖x - x_init‖² + λ · (f(x) - y_target)²",
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
        "tol": args.tol,
        "lambda_start": args.lambda_start,
        "lambda_max": args.lambda_max,
        "lambda_mult": args.lambda_mult,
        "target_delta": args.target_delta,
        "top_k_per_subject": args.top_k,
        "seed": args.seed,
        "elapsed_min": round((time.time() - t0) / 60, 2),
        "git_commit": git_sha(_WORKTREE_ROOT),
        "results": results,
    }
    out_path = out_dir / f"counterfactuals_fold{args.fold}.json"
    out_path.write_text(json.dumps(summary, indent=2))
    logger.info("wrote %s (%.1f min total)", out_path, summary["elapsed_min"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
