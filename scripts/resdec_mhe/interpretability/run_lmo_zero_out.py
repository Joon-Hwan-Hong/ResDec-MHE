"""Leave-Multiple-Out (LMO) ablation via joint zero-out at inference.

Extends ``run_loco_zero_out.py`` (single-CT zero-out) to **joint** zeroing of
CT subsets. Tests whether the top-k most load-bearing or top-k highest-
attribution CTs encode REDUNDANT (super-additive joint ΔR²) or INDEPENDENT
(sub-additive / linear joint ΔR²) signal vs. the sum of their individual
LOCO ΔR²s.

For each combination size ``k ∈ {2, 3, 5, 10}``:

* ``loco_top``     — top-k CTs by canonical LOCO ΔR² (most negative).
* ``captum_top``   — top-k CTs by Captum mean importance (composite-attribution
                     summary, ``cell_types_ranked_by_total_attribution``).
* ``random``       — 5 random subsets of k well-covered CTs (null comparison).

Joint ΔR² = mean over folds of [r2_zeroed(fold) − r2_canonical(fold)].
Sum-of-individual ΔR² = Σ_{ct ∈ subset} loco[ct].delta_r2_vs_canonical
                        (from canonical LOCO JSON).

Interpretation:

* ``super-additive``: joint ΔR² is **more negative** than sum-of-individual
  → CTs encode shared / redundant signal; jointly zeroing exposes additional
  loss the individual zero-outs missed.
* ``sub-additive``  : joint ΔR² is **less negative** than sum-of-individual
  → CTs encode largely independent signal; the model has redundant routes
  through them.
* ``linear``        : joint ΔR² ≈ sum-of-individual.

Outputs ``<out-dir>/lmo_results.json`` with the schema described in the
project plan.
"""
from __future__ import annotations

import argparse
import itertools
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

from src.data.datamodule import CognitiveResilienceDataModule
from src.data.splits import load_splits
from src.training.resdec_lightning_module import ResDecLightningModule
from src.utils.provenance import git_sha, pick_max_r2_ckpt

logger = logging.getLogger(__name__)


def _zero_out_joint_and_predict(
    model: ResDecLightningModule,
    val_batches: list[dict],
    cts_to_zero: tuple[int, ...] | None,
    tabpfn_outer_map: dict[str, float],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Forward over val_batches with all CTs in ``cts_to_zero`` zeroed jointly.

    If ``cts_to_zero is None``, runs canonical (no zero). Returns
    ``(subject_ids, composite_preds, true_y)``.
    """
    sids_all: list[str] = []
    comp_all: list[float] = []
    true_all: list[float] = []
    cts_set = None if cts_to_zero is None else list(cts_to_zero)
    with torch.no_grad():
        for batch in val_batches:
            batch_d = {
                k: v.to(device) if torch.is_tensor(v) else v
                for k, v in batch.items()
            }
            if cts_set is not None:
                for key in ("pseudobulk", "region_pseudobulk"):
                    v = batch_d.get(key)
                    if v is None or not torch.is_tensor(v):
                        continue
                    v_mod = v.clone()
                    if key == "pseudobulk":
                        # shape [B, n_cell_types, n_genes]
                        v_mod[:, cts_set, :] = 0.0
                    else:  # region_pseudobulk: [B, n_regions, n_cell_types, n_genes]
                        v_mod[:, :, cts_set, :] = 0.0
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


def _classify_additivity(
    joint_delta: float, sum_indiv_delta: float, tol: float = 0.0,
) -> str:
    """Classify joint vs sum-of-individual ΔR².

    Both deltas should be negative (zeroing hurts performance). We compare
    their **magnitudes** (i.e., how much zeroing this set hurts):

    * super-additive: |joint| > |sum_indiv| + tol
      → the joint zero-out hurts MORE than the sum of individual hurts
      → CTs encode shared / redundant signal (jointly zeroing exposes
        additional dependency beyond the linear sum).
    * sub-additive  : |joint| < |sum_indiv| − tol
      → the joint zero-out hurts LESS than the sum of individual hurts
      → CTs encode largely independent / overlapping representations
        (the model has redundant routes through them).
    * linear        : |joint| − |sum_indiv| within ±tol.

    Equivalent in raw signed deltas (both negative): joint < sum_indiv − tol
    is super-additive (more negative); joint > sum_indiv + tol is sub.
    """
    # Use signed compare (more negative = larger magnitude when both are negative).
    if joint_delta < sum_indiv_delta - tol:
        return "super-additive"
    if joint_delta > sum_indiv_delta + tol:
        return "sub-additive"
    return "linear"


def _resolve_top_k_loco(loco_path: Path, k: int) -> list[tuple[int, str, float]]:
    """Return ``[(idx, name, delta_r2)]`` for top-k most load-bearing CTs."""
    with loco_path.open() as f:
        d = json.load(f)
    rows = sorted(d["per_cell_type"], key=lambda r: r["delta_r2_vs_canonical"])
    return [
        (r["cell_type_index"], r["cell_type"], r["delta_r2_vs_canonical"])
        for r in rows[:k]
    ]


def _resolve_top_k_captum(
    captum_path: Path, k: int, ct_name_to_index: dict[str, int],
) -> list[tuple[int, str, float]]:
    """Return ``[(idx, name, total_abs_attribution)]`` for top-k by Captum."""
    with captum_path.open() as f:
        d = json.load(f)
    rows = d["cell_types_ranked_by_total_attribution"][:k]
    out: list[tuple[int, str, float]] = []
    for r in rows:
        nm = r["cell_type"]
        if nm not in ct_name_to_index:
            raise KeyError(
                f"Captum CT '{nm}' not found in CELL_TYPE_ORDER",
            )
        out.append((ct_name_to_index[nm], nm, float(r["total_abs_attribution"])))
    return out


def _resolve_random_k(
    well_covered_indices: list[int],
    well_covered_names: list[str],
    k: int,
    n_subsets: int,
    rng: np.random.Generator,
) -> list[list[tuple[int, str]]]:
    """Sample ``n_subsets`` random k-subsets of well-covered CTs (no repeats)."""
    if k > len(well_covered_indices):
        return []
    subsets: list[list[tuple[int, str]]] = []
    seen: set[tuple[int, ...]] = set()
    # Bound attempts to avoid infinite loops on small populations.
    max_attempts = max(50, n_subsets * 20)
    attempts = 0
    while len(subsets) < n_subsets and attempts < max_attempts:
        sel = tuple(sorted(
            int(x) for x in rng.choice(well_covered_indices, size=k, replace=False)
        ))
        if sel in seen:
            attempts += 1
            continue
        seen.add(sel)
        idx_to_name = dict(zip(well_covered_indices, well_covered_names))
        subsets.append([(i, idx_to_name[i]) for i in sel])
        attempts += 1
    return subsets


def _eval_subset_over_folds(
    cts_to_zero: list[int],
    fold_state: list[dict],
) -> tuple[list[float], float]:
    """Evaluate one CT subset across all folds; return per-fold ΔR² and mean ΔR²."""
    delta_per_fold: list[float] = []
    for st in fold_state:
        _, comp, true_y = _zero_out_joint_and_predict(
            st["model"], st["val_batches"], tuple(cts_to_zero),
            st["tabpfn_map"], st["device"],
        )
        r2_z = float(r2_score(true_y, comp))
        delta_per_fold.append(r2_z - st["r2_canonical"])
    return delta_per_fold, float(np.mean(delta_per_fold))


def _build_fold_state(
    fold: int,
    cfg,
    canonical_dir: Path,
    splits_path: Path,
    tabpfn_dir: Path,
    device: torch.device,
) -> dict:
    """Load checkpoint, datamodule, val batches, tabpfn map, and canonical R² for one fold."""
    fold_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    OmegaConf.set_struct(fold_cfg, False)
    fold_cfg.data.fold = fold

    fold_dir = canonical_dir / f"fold{fold}"
    ckpt_path = pick_max_r2_ckpt(fold_dir / "checkpoints")
    logger.info("fold %d: loading %s", fold, ckpt_path.name)

    splits = load_splits(str(splits_path))
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

    val_batches: list[dict] = []
    for batch in dm.val_dataloader():
        val_batches.append(batch)
    tabpfn_map = _load_tabpfn_outer_map(tabpfn_dir, fold)

    # Canonical (no zero) for this fold.
    _, comp, true_y = _zero_out_joint_and_predict(
        model, val_batches, None, tabpfn_map, device,
    )
    r2_canon = float(r2_score(true_y, comp))
    logger.info("fold %d canonical R² = %+.4f", fold, r2_canon)

    return {
        "model": model,
        "val_batches": val_batches,
        "tabpfn_map": tabpfn_map,
        "device": device,
        "r2_canonical": r2_canon,
    }


def _run_one_fold_smoke(
    cts_to_zero: list[int],
    fold_state: dict,
) -> tuple[float, float]:
    """Smoke-only single-fold eval."""
    _, comp, true_y = _zero_out_joint_and_predict(
        fold_state["model"], fold_state["val_batches"],
        tuple(cts_to_zero), fold_state["tabpfn_map"], fold_state["device"],
    )
    r2_z = float(r2_score(true_y, comp))
    return r2_z, r2_z - fold_state["r2_canonical"]


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--config", default="configs/resdec_mhe/canonical.yaml")
    p.add_argument(
        "--pred-root", default="outputs/redesign/p5_canonical_seed42",
        help="Per-fold checkpoint root; expects fold0/checkpoints/ etc.",
    )
    p.add_argument("--tabpfn-dir", default="data/redesign")
    p.add_argument("--splits-path", default="outputs/splits.json")
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--n-cell-types", type=int, default=31)
    p.add_argument(
        "--out-dir",
        default="outputs/redesign/interpretability/lmo_zero_out",
    )
    p.add_argument(
        "--coverage-json",
        default="outputs/redesign/interpretability/ct_coverage_full_cohort.json",
    )
    p.add_argument(
        "--loco-json",
        default="outputs/redesign/interpretability/loco_zero_out/loco_per_celltype.json",
    )
    p.add_argument(
        "--captum-summary-json",
        default="outputs/redesign/interpretability/captum_ig/composite_attribution_summary.json",
    )
    p.add_argument(
        "--ks", default="2,3,5,10",
        help="Comma-separated combination sizes",
    )
    p.add_argument(
        "--n-random", type=int, default=5,
        help="Number of random k-subsets per k (for null baseline)",
    )
    p.add_argument(
        "--additivity-tol", type=float, default=1e-4,
        help="ΔR² tolerance for the linear/sub/super classifier",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--smoke-test", action="store_true",
        help="Smoke-only: k=2 top-loco on fold 0; report wall + ΔR² and exit.",
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

    # Axis-aligned CT names from CELL_TYPE_ORDER (must match LOCO indexing).
    from src.data.constants import CELL_TYPE_ORDER
    ct_names: list[str] = list(CELL_TYPE_ORDER[: args.n_cell_types])
    ct_name_to_index: dict[str, int] = {nm: i for i, nm in enumerate(ct_names)}

    # Load coverage for well-covered random null.
    with Path(args.coverage_json).open() as f:
        cov = json.load(f)
    well_covered_pairs: list[tuple[int, str]] = [
        (ct_name_to_index[nm], nm)
        for nm, info in cov["per_ct"].items()
        if info.get("well_covered") and nm in ct_name_to_index
    ]
    well_covered_indices = [i for i, _ in well_covered_pairs]
    well_covered_names = [n for _, n in well_covered_pairs]
    logger.info(
        "well-covered set: %d CTs (filter for random null)",
        len(well_covered_pairs),
    )

    ks: list[int] = [int(x) for x in args.ks.split(",") if x.strip()]
    rng = np.random.default_rng(args.seed)

    t_start = time.time()

    # Smoke path: fold 0 only, k=2 top-loco.
    if args.smoke_test:
        logger.info("SMOKE TEST: fold 0 only, k=2 top-loco")
        fold_state = _build_fold_state(
            fold=0, cfg=cfg,
            canonical_dir=Path(args.pred_root),
            splits_path=Path(args.splits_path),
            tabpfn_dir=Path(args.tabpfn_dir),
            device=device,
        )
        top2_loco = _resolve_top_k_loco(Path(args.loco_json), 2)
        cts = [t[0] for t in top2_loco]
        names = [t[1] for t in top2_loco]
        r2_z, delta = _run_one_fold_smoke(cts, fold_state)
        wall = time.time() - t_start
        smoke = {
            "k": 2,
            "subset_indices": cts,
            "subset_names": names,
            "fold": 0,
            "r2_canonical_fold0": fold_state["r2_canonical"],
            "r2_zeroed_fold0": r2_z,
            "delta_r2_fold0": delta,
            "wall_seconds": round(wall, 2),
        }
        out_smoke = out_dir / "lmo_smoke.json"
        out_smoke.write_text(json.dumps(smoke, indent=2))
        logger.info(
            "SMOKE: top-2 LOCO = %s ΔR²=%+.4f (canonical fold0 R²=%+.4f, "
            "zeroed=%+.4f); wall=%.1fs; wrote %s",
            names, delta, fold_state["r2_canonical"], r2_z, wall, out_smoke,
        )
        return 0

    # Build all fold states up-front (load each fold once).
    fold_states: list[dict] = [
        _build_fold_state(
            fold=f, cfg=cfg,
            canonical_dir=Path(args.pred_root),
            splits_path=Path(args.splits_path),
            tabpfn_dir=Path(args.tabpfn_dir),
            device=device,
        )
        for f in range(args.n_folds)
    ]
    canonical_per_fold = [s["r2_canonical"] for s in fold_states]
    canon_mean = float(np.mean(canonical_per_fold))

    # Load LOCO JSON once for sum-of-individual ΔR² lookup.
    with Path(args.loco_json).open() as f:
        loco_d = json.load(f)
    indiv_delta_by_idx: dict[int, float] = {
        r["cell_type_index"]: r["delta_r2_vs_canonical"]
        for r in loco_d["per_cell_type"]
    }

    results: dict[str, dict] = {}

    for k in ks:
        logger.info("=== k=%d ===", k)
        bucket: dict[str, object] = {}

        # 1) loco_top
        top_loco = _resolve_top_k_loco(Path(args.loco_json), k)
        cts = [t[0] for t in top_loco]
        names = [t[1] for t in top_loco]
        sum_indiv = float(sum(indiv_delta_by_idx[c] for c in cts))
        delta_per_fold, joint = _eval_subset_over_folds(cts, fold_states)
        loco_top = {
            "subset_indices": cts,
            "subset_names": names,
            "sum_individual_delta_r2": sum_indiv,
            "joint_delta_r2": joint,
            "delta_r2_per_fold": delta_per_fold,
            "delta_r2_std": float(np.std(delta_per_fold, ddof=1)),
            "additivity": _classify_additivity(joint, sum_indiv, args.additivity_tol),
        }
        bucket["loco_top"] = loco_top
        logger.info(
            "  loco_top  : joint=%+.5f  sum_indiv=%+.5f  → %s",
            joint, sum_indiv, loco_top["additivity"],
        )

        # 2) captum_top
        top_cap = _resolve_top_k_captum(
            Path(args.captum_summary_json), k, ct_name_to_index,
        )
        cts = [t[0] for t in top_cap]
        names = [t[1] for t in top_cap]
        sum_indiv = float(sum(indiv_delta_by_idx[c] for c in cts))
        delta_per_fold, joint = _eval_subset_over_folds(cts, fold_states)
        captum_top = {
            "subset_indices": cts,
            "subset_names": names,
            "sum_individual_delta_r2": sum_indiv,
            "joint_delta_r2": joint,
            "delta_r2_per_fold": delta_per_fold,
            "delta_r2_std": float(np.std(delta_per_fold, ddof=1)),
            "additivity": _classify_additivity(joint, sum_indiv, args.additivity_tol),
        }
        bucket["captum_top"] = captum_top
        logger.info(
            "  captum_top: joint=%+.5f  sum_indiv=%+.5f  → %s",
            joint, sum_indiv, captum_top["additivity"],
        )

        # 3) random k well-covered subsets
        random_subsets = _resolve_random_k(
            well_covered_indices, well_covered_names, k, args.n_random, rng,
        )
        random_results: list[dict] = []
        for subset in random_subsets:
            cts = [t[0] for t in subset]
            names = [t[1] for t in subset]
            sum_indiv = float(sum(indiv_delta_by_idx[c] for c in cts))
            delta_per_fold, joint = _eval_subset_over_folds(cts, fold_states)
            random_results.append({
                "subset_indices": cts,
                "subset_names": names,
                "sum_individual_delta_r2": sum_indiv,
                "joint_delta_r2": joint,
                "delta_r2_per_fold": delta_per_fold,
                "delta_r2_std": float(np.std(delta_per_fold, ddof=1)),
                "additivity": _classify_additivity(
                    joint, sum_indiv, args.additivity_tol,
                ),
            })
        if random_results:
            joints = [r["joint_delta_r2"] for r in random_results]
            random_summary = {
                "n_subsets": len(random_results),
                "joint_delta_r2_mean": float(np.mean(joints)),
                "joint_delta_r2_std": (
                    float(np.std(joints, ddof=1)) if len(joints) > 1 else 0.0
                ),
                "joint_delta_r2_min": float(np.min(joints)),
                "joint_delta_r2_max": float(np.max(joints)),
            }
        else:
            random_summary = {
                "n_subsets": 0,
                "note": (
                    f"k={k} > #well-covered ({len(well_covered_indices)}); "
                    "random null skipped."
                ),
            }
        bucket["random"] = random_results
        bucket["random_summary"] = random_summary
        logger.info(
            "  random    : n=%d  joint mean=%s",
            len(random_results),
            random_summary.get("joint_delta_r2_mean", "N/A"),
        )

        results[f"k_{k}"] = bucket

    # Top-level interpretation: the loco-top family is the canonical signal.
    # Pick the most extreme k (max k that has loco_top) for the headline call.
    headline_k = max(ks)
    headline = results[f"k_{headline_k}"]["loco_top"]
    interpretation = (
        f"Headline at k={headline_k} (loco_top): joint ΔR² = "
        f"{headline['joint_delta_r2']:+.5f} vs sum-of-individual = "
        f"{headline['sum_individual_delta_r2']:+.5f} → "
        f"{headline['additivity']}. "
        "Per-k entries report joint ΔR² vs sum-of-individual ΔR² for "
        "loco_top, captum_top, and 5 random well-covered subsets."
    )

    provenance = {
        "canonical_5fold_r2": canonical_per_fold,
        "canonical_mean_r2": canon_mean,
        "canonical_std_r2": float(np.std(canonical_per_fold, ddof=1)),
        "n_folds": args.n_folds,
        "n_cell_types": args.n_cell_types,
        "ks": ks,
        "n_random_per_k": args.n_random,
        "additivity_tol": args.additivity_tol,
        "seed": args.seed,
        "device": str(device),
        "n_well_covered": len(well_covered_indices),
        "loco_json": str(args.loco_json),
        "captum_summary_json": str(args.captum_summary_json),
        "coverage_json": str(args.coverage_json),
        "elapsed_min": round((time.time() - t_start) / 60, 2),
        "git_commit": git_sha(_WORKTREE_ROOT),
    }

    out_payload = {**results, "interpretation": interpretation, "provenance": provenance}
    out_json = out_dir / "lmo_results.json"
    out_json.write_text(json.dumps(out_payload, indent=2))
    logger.info(
        "wrote %s (%.1f min)", out_json, provenance["elapsed_min"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
